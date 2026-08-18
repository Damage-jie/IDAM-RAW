
import torch
import torch.nn as nn
from .components import Matrix_Predictor, Color_Level_Process, WaveDualFusion


class DetISP(nn.Module):
    """
    基于物理感知的 DetISP 网络。
    实现了 RAW(RGGB) -> 自适应白平衡/色彩校正 -> 非线性Gamma映射 -> 双流融合输出 RGB。
    """

    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        # 1. Linear Component: 预测空间仿射矩阵残差 (已移除无用 data 参数)
        self.linear_predictor = Matrix_Predictor(dim=32)

        # 2. Nonlinear Component: 8阶泰勒色调映射
        self.nonlinear = Color_Level_Process(number_f=32)

        # 3. 双流小波融合
        self.fusion = WaveDualFusion(in_dim=3)

        # 全局可学习物理参数 (兜底机制)
        self.default_wb = nn.Parameter(torch.ones(1, 3), requires_grad=True)
        self.default_ccm = nn.Parameter(torch.eye(3).unsqueeze(0), requires_grad=True)

        # 🌟 为 Self-Boost Loss 预留的内部状态存储器
        # 这样不会破坏 YOLO 的 Sequential 结构，又能随时在外部调用
        self.lin_rgb_out = None
        self.nonlin_rgb_out = None

    def forward(self, x):
        B, C, H, W = x.shape

        # ==========================================================
        # 🚨 【前向兼容补丁】 防止 YOLO 初始化测算步长时因 3 通道导致崩溃
        if C == 3:
            x = torch.cat([x, x[:, :1, :, :]], dim=1)
            B, C, H, W = x.shape
        # ==========================================================

        raw_img = x[:, :4, :, :]

        # 始终使用可学习参数 (针对无 meta 标签的 BDD100K 极其有效)
        wb = self.default_wb.expand(B, -1)
        ccm = self.default_ccm.expand(B, -1, -1)

        # ==========================================================
        # 🌟 核心物理运算：将 WB、Binning、CCM 组合成统一的 3x4 矩阵 P
        # ==========================================================

        # 1. 构造 白平衡+降采样(基于 RGGB) 的算子矩阵 M_{bin+wb} [B, 3, 4]
        M_bin_wb = torch.zeros(B, 3, 4, device=raw_img.device, dtype=raw_img.dtype)
        M_bin_wb[:, 0, 0] = wb[:, 0]  # R
        M_bin_wb[:, 1, 1] = 0.5 * wb[:, 1]  # G1
        M_bin_wb[:, 1, 2] = 0.5 * wb[:, 1]  # G2
        M_bin_wb[:, 2, 3] = wb[:, 2]  # B

        # 2. 生成基础物理仿射矩阵 P [B, 3, 4]
        P = torch.bmm(ccm, M_bin_wb)

        # 3. 预测空间偏移量 dmat
        preds = self.linear_predictor(raw_img, P)
        dmat = preds[0]  # shape: [B, 12, H, W]

        # 4. 生成自适应像素级仿射矩阵 P'
        P_expand = P.view(B, 12, 1, 1)
        P_prime = P_expand + dmat  # [B, 12, H, W]
        P_prime = P_prime.view(B, 3, 4, H, W)

        # 5. 应用物理矩阵变换
        raw_expand = raw_img.unsqueeze(1)  # [B, 1, 4, H, W]
        lin_rgb = torch.sum(P_prime * raw_expand, dim=2)  # -> [B, 3, H, W]
        lin_rgb = torch.clamp(lin_rgb, 0, 1)

        # === 记录中间变量 (用于后续写 Self-Boost Loss) ===
        if self.training:
            self.lin_rgb_out = lin_rgb

        # === 非线性组件 (Tone Mapping) ===
        nonlin_rgb = self.nonlinear(lin_rgb)

        if self.training:
            self.nonlin_rgb_out = nonlin_rgb

        # === 双流小波融合 ===
        final_rgb = self.fusion(nonlin_rgb, lin_rgb)

        return final_rgb
