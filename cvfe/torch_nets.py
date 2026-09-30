"""torch 网络与训练（B6/B7，服务器 RTX 3090）。

等变复卷积 trunk：无偏置复卷积 + modReLU（模值激活，相位保持）+ 全局平均池化
  -> 对全局相位旋转严格等变：f(e^{iθ}z) = e^{iθ} f(z)
双通道实值 trunk：同构架构的实卷积（Re/Im 通道独立权重）——不保证等变（对照）。
参数按独立实标量计数并报告精确值。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """无偏置复卷积（等变）。weight: (C_out, C_in, k) complex。"""
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.wr = nn.Parameter(torch.randn(c_out, c_in, k) / np.sqrt(c_in * k))
        self.wi = nn.Parameter(torch.randn(c_out, c_in, k) / np.sqrt(c_in * k))

    def forward(self, z):                      # z: (B, C, L) complex
        zr, zi = z.real, z.imag
        or_ = F.conv1d(zr, self.wr) - F.conv1d(zi, self.wi)
        oi = F.conv1d(zr, self.wi) + F.conv1d(zi, self.wr)
        return torch.complex(or_, oi)


class ModReLU(nn.Module):
    """modReLU: z -> z * relu(|z| + b) / |z|（相位保持 => 等变）。"""
    def __init__(self, c):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, z):
        a = z.abs().clamp_min(1e-8)
        g = F.relu(a + self.b[None, :, None]) / a
        return z * g


class ComplexTrunk(nn.Module):
    """等变复卷积编码器：输入 (B,L) complex -> 输出 (B,C_out) complex。"""
    def __init__(self, ch=(16, 16, 8)):
        super().__init__()
        c1, c2, c3 = ch
        self.conv1 = ComplexConv1d(1, c1, 7)
        self.act1 = ModReLU(c1)
        self.conv2 = ComplexConv1d(c1, c2, 5)
        self.act2 = ModReLU(c2)
        self.conv3 = ComplexConv1d(c2, c3, 3)

    def forward(self, z):                      # (B,L) complex
        h = z[:, None, :]
        h = self.act1(self.conv1(h))
        h = self.act2(self.conv2(h))
        h = self.conv3(h)
        return h.mean(dim=-1)                  # 全局平均池化（线性，保等变）

    def n_real_params(self):
        return sum(p.numel() for p in self.parameters())


class RealTrunk(nn.Module):
    """双通道实值对照编码器：(Re,Im) 2 通道实卷积。"""
    def __init__(self, ch=(16, 16, 8)):        # 每层实输出通道 = 2×复通道数
        super().__init__()
        c1, c2, c3 = ch
        self.net = nn.Sequential(
            nn.Conv1d(2, 2 * c1, 7), nn.ReLU(),
            nn.Conv1d(2 * c1, 2 * c2, 5), nn.ReLU(),
            nn.Conv1d(2 * c2, 2 * c3, 3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, z):                      # (B,L) complex
        x = torch.stack([z.real, z.imag], dim=1)     # (B,2,L)
        return self.net(x).squeeze(-1)               # (B,2C3) real

    def n_real_params(self):
        return sum(p.numel() for p in self.parameters())


def equivariance_error(trunk, z, thetas=(np.pi / 4, np.pi / 2, np.pi), device="cuda"):
    """δ = mean_theta ||f(e^{iθ}z) - e^{iθ}f(z)|| / ||f(z)||。z: (B,L) complex numpy"""
    with torch.no_grad():
        zt = torch.as_tensor(z, dtype=torch.complex64, device=device)
        f0 = trunk(zt)
        errs = []
        for th in thetas:
            rot = torch.as_tensor(np.exp(1j * th), dtype=torch.complex64, device=device)
            fr = trunk(zt * rot)
            num = (fr - f0 * rot).abs().norm(dim=-1)
            den = f0.abs().norm(dim=-1).clamp_min(1e-12)
            errs.append((num / den).mean().item())
    return float(np.mean(errs))
