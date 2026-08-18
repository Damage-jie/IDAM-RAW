# 文件路径: ultralytics/nn/modules/iim.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super().__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        # 初始化随机卷积核
        self.filter = nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.filter)
        for bn in [self.rg_bn, self.gb_bn, self.rb_bn]:
            nn.init.constant_(bn.weight, 0.01)
            nn.init.constant_(bn.bias, 0)

    def mean_constraint(self, kernel):
        # 🌟 YOLA 核心：零均值约束，消除光照分量
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        return (kernel.view(bs, -1) - kernel_mean).view(bs, cin, kw, kh)

    def forward(self, img):
        # 保护性 clamp，防止 log(<=0)
        img = torch.clamp(img, min=1e-6, max=1.0)

        # Log 域变换
        log_img = torch.log(img)

        # 提取 R, G, B
        red_chan = log_img[:, 0:1, :, :]
        green_chan = log_img[:, 1:2, :, :]
        blue_chan = log_img[:, 2:3, :, :]

        # 获取归一化卷积核
        normalized_filter = self.mean_constraint(self.filter)

        # 核心公式：对数域差分卷积
        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rg = self.rg_bn(filt_r1 + filt_g1)

        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_gb = self.gb_bn(filt_g2 + filt_b1)

        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rb = self.rb_bn(filt_r2 + filt_b2)

        # 拼接 (B, 24, H, W)
        out = torch.cat([filt_rg, filt_gb, filt_rb], dim=1)
        return out


class IlluminationDecouplingCore(nn.Module):
    """
    IDAM-RAW 专用光照不变模块
    Args:
        in_channels (int): 输入通道数 (通常是 3, 来自 DetISP 输出的 RGB)
        kernel_nums (int): YOLA 内部卷积核数量
    """

    def __init__(self, in_channels=3, kernel_nums=8):
        super().__init__()
        # 1. 原始特征投影 (保留纹理细节)
        self.feat_projector = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. 光照不变特征提取 (去除光照干扰)
        self.iim = ReflectedConvolution(kernel_nums=kernel_nums)
        # 输出通道数固定为 kernel_nums * 3 = 24

        # 3. 融合层: 24 (Projector) + 24 (IIM) = 48 -> 32 -> in_channels
        self.fuse_net = nn.Sequential(
            nn.Conv2d(24 + 24, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        feat_rgb = self.feat_projector(x)
        feat_iim = self.iim(x)

        # 融合
        concat_feat = torch.cat([feat_rgb, feat_iim], dim=1)
        out = self.fuse_net(concat_feat)

        # 残差连接：原始 RGB + 修正量
        # 同时返回 feat_iim 用于计算 UDA 光照一致性 Loss
        return out + x  # , feat_iim  <-- 暂时只返回图像，如果要算 Loss 需要改这里