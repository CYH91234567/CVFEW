"""B24：相位平均梯度坍缩的数值验证（本地 numpy；theory_gradient_collapse.md）。

P1 原型信号：||E[μ̂_coh]||/||μ|| = g₁(σ_θ)（精确）；||E[μ̂_orb]||/||μ|| = I₁(ρ_a)/I₀(ρ_a)
   （σ_θ 全轴平坦，ρ_a = 2/σ²）。
P2 梯度 SNR：2D 模型、单复特征 w（e(z)=w^Hz），stop-grad 语义 FD 梯度——
   naive 目标 SNR ∝ g₁ 坍缩；L1PCA 目标 SNR 平坦。
输出：04_results/logs/B24_gradient_collapse.json
"""
import json, os, sys, time
import numpy as np
from scipy.special import ive

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def aligned_proto(Z, n_iter=2):
    """L1PCA 对齐原型（投影域）：Z:(...,k)。返回 μ̂:(...,) 。"""
    mu = Z.mean(-1)
    for _ in range(n_iter):
        ip = np.conj(mu)[..., None] * Z
        mu = (Z * np.exp(-1j * np.angle(ip))).mean(-1)
    return mu


def main():
    t0 = time.time()
    res = {"checks": {}}
    sts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.pi]
    sig = 0.5                                   # 固定单样本 SNR：ρ_a = 2/σ² = 8
    E, k = 40000, 5

    # ---- P1：原型信号分解（投影域，μ=1）----
    sig_ratio_coh, sig_ratio_orb, sig_ratio_orb_o, sig_ratio_orb_s = [], [], [], []
    for st in sts:
        r = np.random.RandomState(int(st * 100) + 1)
        th = st * r.randn(E, k) if st > 0 else np.zeros((E, k))
        Z = np.exp(1j * th) + (sig / np.sqrt(2)) * (r.randn(E, k) + 1j * r.randn(E, k))
        mu_coh = Z.mean(-1)
        # oracle 参考（v=μ，定理恒等式本义）
        mu_orb_o = (Z * np.exp(-1j * np.angle(Z))).mean(-1)
        # strict 对齐（3 重启 × 10 迭代，同 B22b）
        mu_orb_s = Z.mean(-1)
        for init in (Z[:, 0], Z[:, min(2, k - 1)]):
            mu_a = init
            for _ in range(10):
                mu_a = (Z * np.exp(-1j * np.angle(np.conj(mu_a)[..., None] * Z))).mean(-1)
            o_a = np.abs(np.conj(mu_a)[..., None] * Z).sum(-1)
            o_b = np.abs(np.conj(mu_orb_s)[..., None] * Z).sum(-1)
            mu_orb_s = np.where(o_a > o_b, mu_a, mu_orb_s)
        mu_orb = aligned_proto(Z)                            # mean-init 2 迭代（B19 原版）
        sig_ratio_coh.append(float(np.abs(mu_coh.mean()).real))
        sig_ratio_orb.append(float(np.abs(mu_orb.mean()).real))
        sig_ratio_orb_o.append(float(np.abs(mu_orb_o.mean()).real))
        sig_ratio_orb_s.append(float(np.abs(mu_orb_s.mean()).real))
    g1 = np.array([np.exp(-0.5 * s ** 2) if s > 0 else 1.0 for s in sts])
    R = ive(1, 2 / sig ** 2) / ive(0, 2 / sig ** 2)
    res["P1"] = {"sts": sts, "coh": sig_ratio_coh, "orb_meaninit": sig_ratio_orb,
                 "orb_oracle": sig_ratio_orb_o, "orb_strict": sig_ratio_orb_s,
                 "g1_pred": g1.tolist(), "R_pred": float(R)}
    res["checks"]["P1_coh_max_dev"] = float(np.max(np.abs(np.array(sig_ratio_coh) - g1)))
    res["checks"]["P1_coh_pass"] = bool(np.max(np.abs(np.array(sig_ratio_coh) - g1)) < 5e-3)
    res["checks"]["P1_orb_oracle_max_dev"] = float(np.max(np.abs(np.array(sig_ratio_orb_o) - R)))
    res["checks"]["P1_orb_pass"] = bool(np.max(np.abs(np.array(sig_ratio_orb_o) - R)) < 5e-3)
    res["checks"]["P1_orb_oracle_flat_range"] = float(max(sig_ratio_orb_o) - min(sig_ratio_orb_o))
    res["checks"]["P1_orb_strict_flat_range"] = float(max(sig_ratio_orb_s) - min(sig_ratio_orb_s))
    res["checks"]["P1_orb_meaninit_flat_range"] = float(max(sig_ratio_orb) - min(sig_ratio_orb))

    # ---- P2：梯度 SNR（2D，单复特征 w；对齐相位+读出相位全冻结，无 kink）----
    m1 = np.array([1.0, 0.0])
    w0 = np.array([1.0, 0.3]) + 1j * np.array([0.1, -0.2])
    fd = 1e-4
    coords = [(0, 1), (0, -1), (1, 1), (1, -1)]

    def feats(w, Zs, Zq):
        return (np.einsum("f,efk->ek", np.conj(w), Zs),
                np.einsum("f,ef->e", np.conj(w), Zq))

    def loss_sg(w, Zs, Zq, phases, read_ph, naive):
        zf_s, zf_q = feats(w, Zs, Zq)
        mu = zf_s.mean(-1) if naive else (zf_s * np.exp(-1j * phases)).mean(-1)
        ip = zf_q * np.conj(mu)
        return -(np.conj(read_ph) * ip).real.mean()

    snr = {"naive": [], "l1pca": []}
    for st in sts:
        r = np.random.RandomState(int(st * 100) + 7)
        th_s = st * r.randn(E, k) if st > 0 else np.zeros((E, k))
        th_q = st * r.randn(E) if st > 0 else np.zeros(E)
        Zs = np.exp(1j * th_s)[..., None] * m1[None, None, :] +             (sig / np.sqrt(2)) * (r.randn(E, k, 2) + 1j * r.randn(E, k, 2))
        Zs = Zs.transpose(0, 2, 1)                               # (E,2,k)
        Zq = np.exp(1j * th_q)[:, None] * m1[None, :] +             (sig / np.sqrt(2)) * (r.randn(E, 2) + 1j * r.randn(E, 2))
        zf_s0, zf_q0 = feats(w0, Zs, Zq)
        for obj in ("naive", "l1pca"):
            if obj == "naive":
                phases = None
                mu0_ = zf_s0.mean(-1)
            else:
                phases = np.angle(np.conj(aligned_proto(zf_s0))[..., None] * zf_s0 + 1e-30)
                mu0_ = (zf_s0 * np.exp(-1j * phases)).mean(-1)
            read_ph = np.angle(zf_q0 * np.conj(mu0_) + 1e-30)
            gsub = []
            for b in range(8):
                sl = slice(b * E // 8, (b + 1) * E // 8)
                gb = np.zeros(4)
                for ci, (comp, sgn) in enumerate(coords):
                    wp = w0.copy(); wp[comp] += sgn * fd
                    wp2 = w0.copy(); wp2[comp] -= sgn * fd
                    lp = loss_sg(wp, Zs[sl], Zq[sl], phases[sl] if phases is not None else None,
                                 read_ph[sl], obj == "naive")
                    lm = loss_sg(wp2, Zs[sl], Zq[sl], phases[sl] if phases is not None else None,
                                 read_ph[sl], obj == "naive")
                    gb[ci] = (lp - lm) / (2 * sgn * fd) / 2
                gsub.append(gb)
            gsub = np.array(gsub)
            sig_n = float(np.linalg.norm(gsub.mean(0)))
            noise_n = float(np.sqrt((gsub.std(0, ddof=1) ** 2).sum()))
            snr[obj].append(sig_n / max(noise_n, 1e-12))
    res["P2"] = {"sts": sts, "snr_naive": snr["naive"], "snr_l1pca": snr["l1pca"]}
    # 坍缩判据：naive SNR 在 π 处 / 在 0 处 ≤ g₁(π)/g₁(0)+容差，且 l1pca 平坦
    sn = np.array(snr["naive"]); sl = np.array(snr["l1pca"])
    res["checks"]["P2_naive_collapse_ratio"] = float(sn[-1] / max(sn[0], 1e-12))
    res["checks"]["P2_naive_pass"] = bool(sn[-1] / max(sn[0], 1e-12) < 1/3)
    res["checks"]["P2_l1pca_flat_range"] = float(sl.max() - sl.min())
    res["checks"]["P2_l1pca_rel_range"] = float((sl.max() - sl.min()) / max(sl.mean(), 1e-12))
    res["checks"]["P2_l1pca_flatter_than_naive"] = bool(
        (sl.max() - sl.min()) / max(sl.mean(), 1e-12)
        < (sn.max() - sn.min()) / max(sn.mean(), 1e-12))

    res["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = os.path.join(BASE, "04_results", "logs", "B24_gradient_collapse.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["checks"], indent=1))
    print(f"B24 -> {out}")


if __name__ == "__main__":
    main()
