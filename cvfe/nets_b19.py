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
    def __init__(self, ch=(32, 64, 128, 32), init="rand", pooling="mean", orb_iter=5,
                 orb_lam=1.0, rdim=128, tau_max=6):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.pooling = pooling
        self.orb_iter = orb_iter
        self.orb_lam = orb_lam
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
        if pooling == "learnlam":
            self.lamg = nn.Parameter(torch.zeros(c4))     # 逐通道对齐强度（σ(0)=0.5 起步）
        self.tau_max = tau_max
        # ---- B32：不变读出的复合升级（不变统计 + 学习度量头，B32-A 复合不变性）----
        # metric/invpow_metric：实化不变统计（+功率）→ 实线性投影 → L2 归一。
        # 线性层作用在**逐点不变**的特征上 ⇒ 复合后嵌入精确不变（构造性证书，δ=0）。
        if pooling in ("metric", "invpow_metric"):
            din = 2 * c4 * (tau_max + 2) + (c4 if pooling == "invpow_metric" else 0)
            self.metric = nn.Linear(din, rdim)
        if pooling == "autocorr":
            # 不变结构读出：R_c(τ) (τ=1..tau_max) + m2, m4 ⇒ C*(tau_max+2) 复向量
            self.out_dim = c4 * (tau_max + 2)
        elif pooling == "invpow":
            self.out_dim = c4 * (tau_max + 2) + c4        # cat(invariant_stats, power)
        elif pooling in ("metric", "invpow_metric"):
            self.out_dim = rdim
        else:
            self.out_dim = 2 * c4 if pooling == "hybrid" else c4

    def invariant_stats(self, h):
        """不变结构读出（B31，量纲无关版）：全局相位精确不变 + 保留时间相位结构。

        R̂_c(τ) = R_c(τ)/m2  ——归一化自相关：|R̂|≤1，相位 = 每样本平均相移
        κ_c = m4/m2²        ——归一化峭度（星座密度/阶数，经典 AMC 累量特征）
        log m2_c            ——绝对能量/SNR 层信息
        m2_c = mean_t|h_c|²，m4_c = mean_t|h_c|⁴。全部满足 h→e^{iθ}h 不变
        （相位差与模值不受影响）⇒ 嵌入精确不变（非等变）；无相干求和 ⇒ 无相消；
        不对齐时间相位 ⇒ 保信息（B30 命题二分法的正解）。
        量纲归一化防止 m4（|h|⁴ 量级）主导范数、压没 R̂(τ) 信息。
        h:(B,C,L) -> (B, C*(tau_max+2)) complex。
        """
        L = h.shape[-1]
        a2 = h.abs() ** 2
        m2 = a2.mean(-1).clamp_min(1e-12)
        feats = []
        for tau in range(1, self.tau_max + 1):
            r = (h[..., tau:] * h[..., : L - tau].conj()).mean(-1)
            feats.append(r / m2)                              # R̂(τ)（O(1) 复值）
        feats.append((a2 ** 2).mean(-1) / (m2 ** 2))          # κ（O(1) 实值）
        feats.append(torch.log(m2))                           # log 能量
        return torch.cat(feats, dim=-1)

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
            return orbit_pool(h, n_iter=self.orb_iter, lam=self.orb_lam)   # 严格等变（分析式同步）
        if self.pooling == "hybrid":
            return torch.cat([orbit_pool(h, n_iter=self.orb_iter, lam=self.orb_lam),
                              h.mean(-1)], dim=-1)
        if self.pooling == "learnlam":
            # 逐通道学习对齐强度：e_c = σ(b_c)·orbit_c + (1−σ(b_c))·mean_c。
            # 两分支严格等变 + 实系数混合 ⇒ 等变（架构层"截断对齐"，B26 Prop(v)）。
            w = torch.sigmoid(self.lamg)[None, :]
            return w * orbit_pool(h, n_iter=self.orb_iter,
                                  lam=self.orb_lam) + (1.0 - w) * h.mean(-1)
        if self.pooling == "autocorr":
            return self.invariant_stats(h)               # 精确不变 + 结构保持（B31）
        if self.pooling == "invpow":                     # 不变双视图：结构 + 幅度
            pw = torch.complex(torch.sqrt((h.abs() ** 2).mean(dim=-1) + 1e-12),
                               torch.zeros_like(h.mean(-1).real))
            return torch.cat([self.invariant_stats(h), pw], dim=-1)
        if self.pooling in ("metric", "invpow_metric"):
            # B32：复合不变性 —— 不变统计（实化）[+功率] → 实线性度量头 → L2
            x = torch.cat([self.invariant_stats(h).real, self.invariant_stats(h).imag], -1)
            if self.pooling == "invpow_metric":
                x = torch.cat([x, torch.sqrt((h.abs() ** 2).mean(dim=-1) + 1e-12)], -1)
            return torch.nn.functional.normalize(self.metric(x), dim=-1)
        if self.pooling == "nopool":
            return h.reshape(h.shape[0], -1)             # 序列拉平（无池化=无相消）
        # gated：门仅依赖 |h|（不变标量）⇒ e = Σ a_t h_t 严格等变
        logits = self.beta[None, :, None] * torch.log(h.abs() + 1e-6)
        a = torch.softmax(logits, dim=-1)
        return (a * h).sum(dim=-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def invariance_error(trunk, z, thetas=(np.pi / 4, np.pi / 2, np.pi, 2.0), device="cuda"):
    """不变证书（B32）：δ_inv = mean_θ ||f(e^{iθ}z) − f(z)|| / ||f(z)||。
    不变读出（autocorr/metric/...）应给出 float 级 δ_inv≈1e-7（构造性不变）；
    等变读出（mean/gated/orbit）此处 ≈|1−e^{iθ}|≈1，它们用 equivariance_error。"""
    with torch.no_grad():
        zt = torch.as_tensor(z, dtype=torch.complex64, device=device)
        f0 = trunk(zt)
        errs = []
        for th in thetas:
            rot = torch.as_tensor(np.exp(1j * th), dtype=torch.complex64, device=device)
            fr = trunk(zt * rot)
            num = (fr - f0).abs().norm(dim=-1)
            den = f0.abs().norm(dim=-1).clamp_min(1e-12)
            errs.append((num / den).mean().item())
    return float(np.mean(errs))


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
