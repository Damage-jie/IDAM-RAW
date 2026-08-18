import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm for small-batch stability."""
    g = min(max_groups, num_channels)
    # 🌟 核心修复：强制每个 group 至少分配 2 个通道！
    # 否则当遇到 AdaptiveAvgPool2d(1) 产生的 1x1 特征图时，无法计算方差。
    while g > 1 and (num_channels % g != 0 or num_channels // g < 2):
        g -= 1
    g = max(1, g)  # 兜底保护
    return nn.GroupNorm(g, num_channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0,
                 act: bool = True, gn_groups: int = None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        if gn_groups is None:
            self.gn = _gn(out_ch)
        else:
            self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class SeverityEstimator(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(inplace=False),
            nn.Linear(hidden, 1),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean_abs = xf.abs().mean(dim=(1, 2, 3))
        var = xf.var(dim=(1, 2, 3), unbiased=False)
        feats = torch.stack([torch.log1p(mean_abs), torch.log1p(var)], dim=1)
        a = torch.sigmoid(self.mlp(feats))
        return a.view(-1, 1, 1, 1)


class AdaptiveGate(nn.Module):
    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        mid = max(1, channels // r)
        self.local = nn.Sequential(
            ConvGNAct(channels, mid, 1, 1, 0, act=True),
            ConvGNAct(mid, channels, 1, 1, 0, act=False),
        )
        self.global_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNAct(channels, mid, 1, 1, 0, act=True),
            ConvGNAct(mid, channels, 1, 1, 0, act=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Zero-init last convs
        if isinstance(self.local[-1], ConvGNAct) and isinstance(self.local[-1].conv, nn.Conv2d):
            nn.init.zeros_(self.local[-1].conv.weight)
        if isinstance(self.global_[-1], ConvGNAct) and isinstance(self.global_[-1].conv, nn.Conv2d):
            nn.init.zeros_(self.global_[-1].conv.weight)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        logit = self.local(x) + self.global_(x)
        gate = self.sigmoid(logit)
        return x * (1.0 + alpha * (2.0 * gate - 1.0))


class AFMMFPN(nn.Module):
    """ALP-FPN adapted for YOLOv8."""

    def __init__(self, c1, c2, severity_hidden=32, gate_reduction=4):
        """
        c1: list of input channels (e.g., [256, 512, 1024])
        c2: output channels (e.g., 256)
        """
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2

        # severity estimator
        self.severity = SeverityEstimator(hidden=severity_hidden)

        # lateral & fpn convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # HFP-like gates on laterals
        self.hfp_gates = nn.ModuleList()

        for ch in self.in_channels:
            self.lateral_convs.append(ConvGNAct(ch, c2, 1, 1, 0, act=True))
            self.hfp_gates.append(AdaptiveGate(c2, r=gate_reduction))
            self.fpn_convs.append(ConvGNAct(c2, c2, 3, 1, 1, act=True))

        # SEA-like gates on outputs
        self.sea_gates = nn.ModuleList([
            AdaptiveGate(c2, r=gate_reduction) for _ in range(len(c1))
        ])

    def forward(self, inputs):
        # inputs is a list [P3, P4, P5]

        # Estimate severity from the first input (highest resolution, P3)
        sev_src = inputs[0]
        alpha = self.severity(sev_src)

        # build laterals with HFP gates
        laterals = []
        for i, l_conv in enumerate(self.lateral_convs):
            x = l_conv(inputs[i])
            x = self.hfp_gates[i](x, alpha)
            laterals.append(x)

        # top-down fusion (Last -> First)
        # standard FPN: P5 -> P4 -> P3
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample laterals[i] and add to laterals[i-1]
            target_size = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=target_size, mode='nearest')

        # outputs with FPN convs
        outs = []
        for i in range(len(laterals)):
            outs.append(self.fpn_convs[i](laterals[i]))

        # SEA-like gates on outputs
        final_outs = []
        for i, out in enumerate(outs):
            final_outs.append(self.sea_gates[i](out, alpha))

        return final_outs

class AFMM(nn.Module):
    """
    单层特征净化器 (从 AFMMFPN 中无损剥离)
    拦截特征图，计算全局严重程度(Severity)，并利用自适应门控(AdaptiveGate)滤除噪点。
    """
    def __init__(self, c1, c2, hidden=32, r=4):
        super().__init__()
        # 复用原版 AFMMFPN 最核心的两个算子
        self.severity = SeverityEstimator(hidden=hidden)
        self.gate = AdaptiveGate(c1, r=r)
        # 如果通道数发生变化（通常保持不变），用 1x1 卷积对齐
        self.cv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 1. 评估当前特征图的噪点严重程度
        alpha = self.severity(x)
        # 2. 激活门控过滤噪点，并输出纯净特征
        out = self.gate(x, alpha)
        return self.cv(out)