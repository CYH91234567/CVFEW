"""B17 编码器：容量匹配的等变复 CNN 与实值对照（复/实同嵌入宽度）。

设计要点（等变性纪律）：
  - 复卷积无偏置（偏置破坏等变：b 不随 e^{iθ} 旋转）；
  - ModBN：z / sqrt(E|z|²) * gain —— 统计量取模值（相位不变）、缩放为正实数 ⇒ 严格等变；
    不加平移（复平移破坏等变）。
  - 池化用均值池化（线性 ⇒ 等变）；不用 max-pool（复值 max 无定义/破坏等变）。
  - 输出为复向量，U(1) 作用 = 逐元素乘 e^{iθ}，商空间几何（轨道/PhaseMAP）可直接套用。

对照 RealAMC：同架构宽度（复通道 c ↔ 实通道 2c）、同嵌入维度（32 复 = 64 实），
参数量约为复版的 2×（复卷积以一半实自由度实现同宽映射——如实报告）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """无偏置复卷积（严格等变）。weight = wr + i·wi。"""
    def __init__(self, c_in, c_out, k, padding=0):
        super().__init__()
        self.wr = nn.Parameter(torch.randn(c_out, c_in, k) / np.sqrt(c_in * k))
        self.wi = nn.Parameter(torch.randn(c_out, c_in, k) / np.sqrt(c_in * k))
        self.padding = padding

    def forward(self, z):                      # z:(B,C,L) complex
        zr, zi = z.real, z.imag
        or_ = F.conv1d(zr, self.wr, padding=self.padding) - F.conv1d(zi, self.wi, padding=self.padding)
        oi = F.conv1d(zr, self.wi, padding=self.padding) + F.conv1d(zi, self.wr, padding=self.padding)
        return torch.complex(or_, oi)


class ModBN1d(nn.Module):
    """模值批归一化：z -> z·gain / sqrt(E|z|²)。严格等变（正实缩放）。"""
    def __init__(self, c, eps=1e-4, momentum=0.05):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(c))
        self.register_buffer("running", torch.ones(c))
        self.eps, self.mom = eps, momentum

    def forward(self, z):                      # (B,C,L) complex
        if self.training:
            ms = z.abs().pow(2).mean(dim=(0, 2)).detach()
            self.running.mul_(1.0 - self.mom).add_(self.mom * ms)
            s = torch.rsqrt(ms + self.eps)
        else:
            s = torch.rsqrt(self.running + self.eps)
        return z * (s * self.gain)[None, :, None]


class ModReLU(nn.Module):
    """modReLU: z·relu(|z|+b)/|z|（相位保持 ⇒ 等变）。"""
    def __init__(self, c):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, z):
        a = z.abs().clamp_min(1e-8)
        return z * (F.relu(a + self.b[None, :, None]) / a)


def cpool2(z):
    """长度减半均值池化（线性 ⇒ 等变）。"""
    B, C, L = z.shape
    return z.reshape(B, C, L // 2, 2).mean(-1)


class ComplexAMC(nn.Module):
    """等变复 AMC 编码器：(B,L) complex -> (B,c4) complex。"""
    def __init__(self, ch=(32, 64, 128, 32)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.c1 = ComplexConv1d(1, c1, 7, padding=3); self.b1 = ModBN1d(c1); self.a1 = ModReLU(c1)
        self.c2 = ComplexConv1d(c1, c2, 5, padding=2); self.b2 = ModBN1d(c2); self.a2 = ModReLU(c2)
        self.c3 = ComplexConv1d(c2, c3, 3, padding=1); self.b3 = ModBN1d(c3); self.a3 = ModReLU(c3)
        self.c4 = ComplexConv1d(c3, c4, 3, padding=1); self.b4 = ModBN1d(c4)
        self.out_dim = c4

    def forward(self, z):                      # (B,L) complex
        h = z[:, None, :]
        h = cpool2(self.a1(self.b1(self.c1(h))))
        h = cpool2(self.a2(self.b2(self.c2(h))))
        h = cpool2(self.a3(self.b3(self.c3(h))))
        h = self.b4(self.c4(h))
        return h.mean(dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class RealAMC(nn.Module):
    """实值对照（非等变）：2 通道 (Re,Im) 输入 -> 64 实维输出（可按 view_as_complex 视为 32 复维）。"""
    def __init__(self, ch=(64, 128, 256, 64)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.net = nn.Sequential(
            nn.Conv1d(2, c1, 7, padding=3), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, 3, padding=1), nn.BatchNorm1d(c3), nn.ReLU(),
            nn.Conv1d(c3, c4, 3, padding=1), nn.BatchNorm1d(c4),
        )
        self.out_dim = c4 // 2                # 复视角通道数

    def forward(self, z):                      # (B,L) complex -> (B,out_dim) complex
        x = torch.stack([z.real, z.imag], dim=1)
        h = self.net(x).mean(dim=-1)           # (B,64) real
        return torch.view_as_complex(h.reshape(h.shape[0], -1, 2))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def unit_norm(e):
    return e / e.abs().norm(dim=-1, keepdim=True).clamp_min(1e-8)


def equivariance_error(trunk, z, thetas=(np.pi / 4, np.pi / 2, np.pi, 2.0), device="cuda"):
    """δ = mean_θ ||f(e^{iθ}z) − e^{iθ}f(z)|| / ||f(z)||。"""
    with torch.no_grad():
        zt = torch.as_tensor(z, dtype=torch.complex64, device=device)
        f0 = trunk(zt)
        errs = []
        for th in thetas:
            rot = torch.as_tensor(np.exp(1j * th), dtype=torch.complex64, device=device)
            num = (trunk(zt * rot) - f0 * rot).abs().norm(dim=-1)
            den = f0.abs().norm(dim=-1).clamp_min(1e-12)
            errs.append((num / den).mean().item())
    return float(np.mean(errs))
