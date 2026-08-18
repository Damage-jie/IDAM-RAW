
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .illumination_core import IlluminationDecouplingCore  # 确保导入了你原来的 IlluminationDecouplingCore


class RIDM(nn.Module):
    """
    纯串联结构的安全包装器：
    将 IIM 转变为残差学习，并使用零初始化，保护前端 DetISP 不受冷启动梯度雪崩的影响。
    """

    def __init__(self, c1=3, kernel_nums=8):
        super().__init__()
        # 内部依然是你的原版 IIM，完全不用改它
        self.iim = IlluminationDecouplingCore(in_channels=c1, kernel_nums=kernel_nums)

        # 🌟 核心魔法：可学习的残差权重，初始值强行设为 0！
        # 这使得训练第一步时 out = x，IIM 完全透明，充当梯度高速公路
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 网络通过反向传播，自己决定应该混入多少 IIM 的提亮效果
        out = x + self.alpha * self.iim(x)
        return out.clamp(0, 1)



# ================= 工具函数 =================
def gaussian_kernel(size: int, sigma: float):
    """生成高斯卷积核"""
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g /= g.sum()
    return torch.from_numpy(g).float()


# ================= 线性组件 (预测物理矩阵残差) =================
class GaussianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        kernel = gaussian_kernel(kernel_size, sigma)
        # 扩展为卷积所需的形状 [out, in, k, k]
        self.weight = nn.Parameter(data=kernel.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=bias)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class FA(nn.Module):
    """Local Feature Attention 模块"""

    def __init__(self, channels, elements=12):
        super().__init__()
        self.elements = elements
        self.q = nn.Conv2d(channels, elements, 1, 1)
        self.proj = nn.Sequential(nn.Linear(1, 32), nn.Linear(32, 8))
        self.k = nn.Linear(1, channels)
        self.v = nn.Linear(1, channels)
        self.ada_drop = nn.Dropout(0.1)
        self.lc_drop = nn.Dropout(0.1)

    def forward(self, x, q_input):
        q_input = q_input.flatten(1).unsqueeze(2)  # [B, 12, 1]
        q = self.q(x)
        k = self.k(q_input)
        v = self.v(q_input)

        q_ = q.unsqueeze(4)  # [B, 12, H, W, 1]
        k_ = k.unsqueeze(2).unsqueeze(2)  # [B, 12, 1, 1, C]
        v_ = v.unsqueeze(2).unsqueeze(2)  # [B, 12, 1, 1, C]

        ada_weights = torch.sum(q_ * k_, dim=1, keepdim=True) / (self.elements ** 0.5)
        ada_weights = ada_weights.softmax(dim=4)
        ada_weights = self.ada_drop(ada_weights)

        local = (v_ * ada_weights).sum(dim=4)
        local = local.unsqueeze(-1)
        local = self.proj(local)
        local = self.lc_drop(local)
        return local.sum(dim=-1)


class Matrix_Predictor(nn.Module):
    """预测 12 通道空间仿射变换偏移量的核心网络"""

    def __init__(self, dim=32, num_heads=1, down_drop=0.2):
        super().__init__()

        # 标准卷积流
        def lc_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.Conv2d(out_c, out_c, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.GELU(),
                nn.Dropout(down_drop)
            )

        # 高斯卷积流 (🌟 显式指定 padding=1 避免维度崩塌)
        def gl_block(in_c, out_c):
            return nn.Sequential(
                GaussianConv2d(in_c, out_c, kernel_size=3, sigma=1, stride=1, padding=1, bias=False),
                nn.Conv2d(out_c, out_c, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.GELU(),
                nn.Dropout(down_drop)
            )

        self.lc_block1 = lc_block(4, dim)
        self.lc_block2 = lc_block(dim, dim)
        self.lc_block3 = lc_block(dim, dim)
        self.lc_block4 = lc_block(dim, dim)
        self.lc_block5 = lc_block(dim, dim)

        self.gl_block1 = gl_block(4, dim)
        self.gl_block2 = gl_block(dim, dim)
        self.gl_block3 = gl_block(dim, dim)
        self.gl_block4 = gl_block(dim, dim)
        self.gl_block5 = gl_block(dim, dim)

        self.feature_atten1 = FA(channels=dim)

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(1, dim)
        self.pooling = nn.AvgPool2d(16, 16)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.down = nn.Linear(dim, 1)

    def forward(self, x, mat):
        # 1. 密集残差特征提取
        l_x1 = self.lc_block1(x)
        l_x2 = self.lc_block2(l_x1)
        l_x3 = self.lc_block3(l_x2)
        l_x4 = self.lc_block4(l_x3)
        d_x1 = self.lc_block5(l_x4 + l_x3 + l_x2 + l_x1)

        g_x1 = self.gl_block1(x)
        g_x2 = self.gl_block2(g_x1)
        g_x3 = self.gl_block3(g_x2)
        g_x4 = self.gl_block4(g_x3)
        d_x2 = self.gl_block5(g_x4 + g_x3 + g_x2 + g_x1)

        # 2. Local 特征
        local = self.feature_atten1(d_x2, mat)

        # 3. Global 特征 (Transformer 风格)
        d_x = self.pooling(d_x1).flatten(2).transpose(1, 2)
        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q_input = mat.flatten(1).unsqueeze(2)  # [B, 12, 1]
        q = torch.sigmoid(self.q(q_input.unsqueeze(2)))
        q = q.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        glo = (attn @ v).transpose(1, 2).reshape(B, 12, C)
        glo = self.proj(glo)
        glo = self.proj_drop(glo)

        glo = self.down(glo).squeeze(2)  # [B, 12]
        glo = glo.unsqueeze(2).unsqueeze(2)  # [B, 12, 1, 1]

        # 4. 全局与局部特征融合
        out = glo + local  # [B, 12, H, W]

        return out, glo, local


# ================= 非线性组件 (色调映射) =================
def Tayler_res(img: torch.Tensor, weights: list) -> torch.Tensor:
    """8阶泰勒非线性映射函数"""
    img = img.clip(0, 1)
    outs = []
    for i in range(3):
        x = img[:, [i], :, :]
        embed = torch.cat([
            torch.zeros_like(x),
            (x - x ** 2),
            (x ** 3) - 3 * (x ** 2) + 2 * x,
            -(x ** 4) + 4 * x ** 3 - 6 * x ** 2 + 3 * x,
            -6 * (x ** 5) + 15 * x ** 4 - 10 * x ** 3 + x,
            -1.2 * (x ** 6) + 3.0 * x ** 5 - 2.4 * x ** 4 + 0.6 * x,
            -1.344 * x ** 7 + 3.36 * x ** 6 - 2.688 * x ** 5 + 0.672 * x,
            -12 * x ** 8 + 48 * x ** 7 - 60 * x ** 6 + 24 * x ** 5,
        ], dim=1)
        out = (embed * weights[i]).sum(dim=1, keepdim=True)
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    return outs


class Color_Level_Process(nn.Module):
    """自适应非线性亮度/对比度调节器"""

    def __init__(self, number_f=32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgs):
        x1 = self.relu(self.e_conv1(imgs))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x_r = self.e_conv5(x1 + x2 + x3 + x4)

        # 利用 Softmax 提供天然稳定初始化的多项式系数
        r1, r2, r3 = torch.split(x_r, 8, dim=1)
        r1 = r1.softmax(dim=1)
        r2 = r2.softmax(dim=1)
        r3 = r3.softmax(dim=1)

        outs = Tayler_res(imgs, [r1, r2, r3])
        imgs = (imgs + outs).clip(0, 1)
        return imgs


# ================= 小波融合组件 =================
def haar_dwt2d(x):
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x11 - x01 - x10) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 + x10 - x01 - x11) * 0.5
    return ll, lh, hl, hh


def haar_idwt2d(ll, lh, hl, hh):
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll - lh - hl + hh) * 0.5
    x11 = (ll + lh - hl - hh) * 0.5
    B, C, H, W = ll.shape
    out = torch.zeros(B, C, H * 2, W * 2, device=ll.device)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


class WaveDualFusion(nn.Module):
    """基于 Haar 小波的双流信息融合模块"""

    def __init__(self, in_dim=3, embed_dim=32):
        super().__init__()
        self.denoise_net = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(embed_dim, in_dim, 3, 1, 1)
        )
        self.denoise_scale = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, in_dim, 1)
        )

    def forward(self, x_tex, x_raw_feat):
        noise = self.denoise_net(x_raw_feat)
        x_clean = x_tex + torch.tanh(self.denoise_scale) * noise

        ll_t, lh_t, hl_t, hh_t = haar_dwt2d(x_tex)
        ll_c, lh_c, hl_c, hh_c = haar_dwt2d(x_clean)

        alpha = torch.sigmoid(self.gate(torch.cat([ll_t, ll_c], 1)))

        ll = alpha * ll_t + (1 - alpha) * ll_c
        lh, hl, hh = lh_t, hl_t, hh_t  # 保留纹理丰富的高频
        out = haar_idwt2d(ll, lh, hl, hh)
        return out
