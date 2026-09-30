"""B19 编码器：可配置初始化与池化的严格等变复 CNN（修复 B17 训练失败）。

根因假设（PREREG_B19 H-a）：随机宽频卷积核 → 末层隐层 h_c(t) 相位时间近均匀 →
均值池化相干求和相消 → 嵌入信噪比≈0 → 梯度纯噪声 → loss 停在 ln5。

修复臂：
  init="hann"   卷积核乘 Hann 窗（低通初始化）→ 隐层时间持久 → 池化不相消（严格等变保持）
  pooling="power"  e_c = √mean_t|h_c(t)|²（不变实嵌入，无相位求和，折中臂）
  pooling="gated"  a_t ∝ exp(β_c·log|h_c(t)|)（不变标量门 ⇒ 等变保持；β init 0 = 均值）
等变性纪律同 nets_b17：无偏置复卷积、ModBN 模值归一、modReLU、线性/门控池化。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvfe.nets_b17 import ComplexConv1d, ModBN1d, ModReLU, cpool2


def orbit_pool(h, n_iter=5, lam=1.0):
    """时间轴 L1-PCA（轨道）池化：e_c = argmax_{|u|=1} Σ_t |h_c(t)^H u|。

    坐标上升（与 estim.orbital_align / run_b19.l1pca_proto 严格同构，样本轴=时间轴）：
      u <- (1/L) Σ_t h_c(t)·e^{−i·arg(u^H h_c(t))}，对齐相位 stop-gradient。
    严格等变：h→e^{iθ}h ⇒ u→e^{iθ}u（对齐角 θ-无关）。
    lam<1 为 B26 Prop(v) 的"截断对齐"插值：只旋转错配角的 λ 份（lam=0 退化为均值）。
    h:(B,C,L) complex -> (B,C) complex。
    """
    u = h.mean(-1)
    for _ in range(n_iter):
        ip = u[..., None].conj() * h                                      # (B,C,L)
        w = torch.exp(-1j * lam * torch.angle(ip)).detach()
        u = (h * w).mean(-1)
    return u


class ComplexAMC2(nn.Module):
    def __init__(self, ch=(32, 64, 128, 32), init="rand", pooling="mean", orb_iter=5):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.pooling = pooling
        self.orb_iter = orb_iter
        self.c1 = ComplexConv1d(1, c1, 7, padding=3); self.b1 = ModBN1d(c1); self.a1 = ModReLU(c1)
        self.c2 = ComplexConv1d(c1, c2, 5, padding=2); self.b2 = ModBN1d(c2); self.a2 = ModReLU(c2)
        self.c3 = ComplexConv1d(c2, c3, 3, padding=1); self.b3 = ModBN1d(c3); self.a3 = ModReLU(c3)
        self.c4 = ComplexConv1d(c3, c4, 3, padding=1); self.b4 = ModBN1d(c4)
        if init == "hann":
            for conv in (self.c1, self.c2, self.c3, self.c4):
                k = conv.wr.shape[-1]
                w = torch.as_tensor(np.hanning(k), dtype=torch.float32)
                conv.wr.data = conv.wr.data * w[None, None, :]
                conv.wi.data = conv.wi.data * w[None, None, :]
        if pooling == "gated":
            self.beta = nn.Parameter(torch.zeros(c4))
        self.out_dim = 2 * c4 if pooling == "hybrid" else c4

    def hidden(self, z):                       # (B,L) complex -> (B,c4,L16)
        h = z[:, None, :]
        h = cpool2(self.a1(self.b1(self.c1(h))))
        h = cpool2(self.a2(self.b2(self.c2(h))))
        h = cpool2(self.a3(self.b3(self.c3(h))))
        return self.b4(self.c4(h))

    def forward(self, z):
        h = self.hidden(z)
        if self.pooling == "mean":
            return h.mean(dim=-1)
        if self.pooling == "power":
            return torch.sqrt((h.abs() ** 2).mean(dim=-1) + 1e-12)      # 实张量
        if self.pooling == "orbit":
            return orbit_pool(h, n_iter=self.orb_iter)                   # 严格等变（分析式同步）
        if self.pooling == "hybrid":
            return torch.cat([orbit_pool(h, n_iter=self.orb_iter), h.mean(-1)], dim=-1)
        # gated：门仅依赖 |h|（不变标量）⇒ e = Σ a_t h_t 严格等变
        logits = self.beta[None, :, None] * torch.log(h.abs() + 1e-6)
        a = torch.softmax(logits, dim=-1)
        return (a * h).sum(dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def cancellation_ratio(trunk, z):
    """末层隐层的池化相消比 |mean_t h_c| / mean_t|h_c|（逐帧逐通道平均）。"""
    h = trunk.hidden(torch.as_tensor(z, dtype=torch.complex64,
                                     device=next(trunk.parameters()).device))
    return float((h.mean(-1).abs() / h.abs().mean(-1).clamp_min(1e-12)).mean().item())


@torch.no_grad()
def fisher_ratio(emb, y):
    """单位化嵌入 tr(类间)/tr(类内)。emb:(N,C) numpy（复或实），y:(N,)。"""
    e = emb.astype(np.complex128)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
    mu_all = e.mean(0)
    num = den = 0.0
    for c in np.unique(y):
        Ec = e[y == c]
        muc = Ec.mean(0)
        num += len(Ec) * float((np.abs(muc - mu_all) ** 2).sum())
        den += float((np.abs(Ec - muc) ** 2).sum())
    return num / max(den, 1e-12)
