"""B13：排序定理组的数值验证（本地numpy）。

C1  Thm2: 均匀相位下欧氏 MC = 1/2（精确性核对，误差应为纯MC噪声）
C2  Thm4: 交叉点 sigma_theta*(rho, sigma) 网格（P_eucl数值积分 vs P_orb非中心卡方1D积分）
C3  Prop5: 线性 regime 公式 P_eucl ≈ 1/2 - (a/√(2π))e^{-σ_θ²/2}
C4  单调性: P_eucl(σ_θ) 网格单调性 + a* 边界
C5  Prop6: K-shot 原型角度方差地板（欧氏 vs oracle对齐轨道）
"""
import json, os, sys
import numpy as np
from scipy.stats import norm, ncx2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_b4 import risk_euclid_numeric

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))


def p_orb_numeric(rho, sigma, n_grid=2048):
    """轨道1-shot成对错分率（population原型，确定性1D积分）。
    A1 = 1 + σζ₁, |A1|² ~ (σ²/2)χ'²₂(nc=2/σ²)；A2|A1 ~ CN(ρA1, σ²(1-ρ²))。
    P(|A2|>|A1|) = E_r[ ncx2.sf( 2r²/(σ²(1-ρ²)), 2, nc=2ρ²r²/(σ²(1-ρ²)) ) ]。"""
    s2 = sigma ** 2 * (1 - rho ** 2)
    # r² = |A1|² 的密度: (σ²/2)·χ'²₂(df=2, nc=2/σ²)
    x = np.linspace(1e-9, 1, 1)  # 占位
    # χ'²₂ 的支撑 [0,∞)：用分位网格
    q = np.linspace(1e-6, 1 - 1e-6, n_grid)
    r2 = (sigma ** 2 / 2) * ncx2.ppf(q, 2, nc=2 / sigma ** 2)
    pdf = ncx2.pdf(2 * r2 / sigma ** 2, 2, nc=2 / sigma ** 2) * 2 / sigma ** 2
    t = 2 * r2 / s2
    nc = 2 * (rho ** 2) * r2 / s2
    g = ncx2.sf(t, 2, nc=nc)
    return float(np.trapezoid(g * pdf, r2))


def c1_thm2(n_mc=4_000_000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for rho, sigma in [(0.3, 0.3), (0.7, 0.2), (0.0, 0.5)]:
        th = rng.uniform(-np.pi, np.pi, n_mc)
        mu1 = np.array([1.0, 0.0], complex)
        mu2 = np.array([rho, np.sqrt(1 - rho ** 2)], complex)
        eta = (rng.randn(n_mc, 2) + 1j * rng.randn(n_mc, 2)) * (sigma / np.sqrt(2))
        z = np.exp(1j * th)[:, None] * mu1[None, :] + eta
        d_eu = (np.abs(z - mu1) ** 2).sum(1) - (np.abs(z - mu2) ** 2).sum(1)
        pe = float((d_eu > 0).mean())
        se = float(np.sqrt(pe * (1 - pe) / n_mc))
        out.append({"rho": rho, "sigma": sigma, "p_euclid_uniform": pe, "mc_se": se,
                    "dev_from_half": abs(pe - 0.5)})
        print(f"[C1] rho={rho} sigma={sigma}: P_eucl(uniform θ)={pe:.4f} ±{se:.4f} "
              f"(dev {abs(pe-0.5):.4f})", flush=True)
    json.dump(out, open(os.path.join(OUT, "logs", "B13_thm2.json"), "w"), indent=1)


def c2_crossing(rho_grid=(0.0, 0.3, 0.7), sig_grid=(0.1, 0.2, 0.3, 0.5, 0.8)):
    st_grid = np.linspace(0.0, np.pi, 25)
    rows = []
    for rho in rho_grid:
        for sigma in sig_grid:
            po = p_orb_numeric(rho, sigma)
            pes = [risk_euclid_numeric(np.array([1.0, 0.0], complex),
                                       np.array([rho, np.sqrt(1 - rho ** 2)], complex),
                                       st, sigma) for st in st_grid]
            cross = None
            for i in range(len(st_grid) - 1):
                if (pes[i] - po) * (pes[i + 1] - po) <= 0 and abs(pes[i] - po) > 1e-12:
                    # 线性插值
                    frac = (po - pes[i]) / (pes[i + 1] - pes[i])
                    cross = float(st_grid[i] + frac * (st_grid[i + 1] - st_grid[i]))
                    break
            rows.append({"rho": rho, "sigma": sigma, "p_orb": po,
                         "p_eucl_at_0": pes[0], "p_eucl_at_pi": pes[-1],
                         "crossing_sigma_th": cross,
                         "orbital_better_everywhere": pes[-1] <= po and pes[0] > po})
            print(f"[C2] rho={rho} sigma={sigma}: P_orb={po:.4f} P_eucl[0]={pes[0]:.4f} "
                  f"P_eucl[pi]={pes[-1]:.4f} crossing={cross}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "logs", "B13_crossing.json"), "w"), indent=1)
    n_cross = sum(1 for r in rows if r["crossing_sigma_th"] is not None)
    print(f"[C2] 交叉点存在: {n_cross}/{len(rows)}; 全程轨道更优: "
          f"{sum(1 for r in rows if r['orbital_better_everywhere'])}/{len(rows)}")


def c3_linear(rho=0.3, sigma=0.6, st_grid=None):
    """线性 regime: P_eucl ≈ 1/2 - (a/√(2π))e^{-σ_θ²/2} + O(a³)。
    验证：把 P_eucl + (a/√(2π))e^{-σ_θ²/2} 的 σ_θ 残差 vs O(a³) 上界。"""
    st_grid = st_grid or np.linspace(0.0, np.pi, 13)
    a = np.sqrt(1 - rho) / sigma
    mu1 = np.array([1.0, 0.0], complex)
    mu2 = np.array([rho, np.sqrt(1 - rho ** 2)], complex)
    rows = []
    for st in st_grid:
        pe = risk_euclid_numeric(mu1, mu2, st, sigma)
        pred = 0.5 - a / np.sqrt(2 * np.pi) * np.exp(-st ** 2 / 2)
        rows.append({"sigma_th": st, "p_euclid": pe, "linear_pred": pred,
                     "resid": pe - pred})
        print(f"[C3] st={st:.2f}: P={pe:.4f} linear={pred:.4f} resid={pe-pred:+.4f}", flush=True)
    print(f"[C3] a={a:.3f} (线性regime要求 a 小)；|resid| max = "
          f"{max(abs(r['resid']) for r in rows):.4f}，应 ≈ O(a³) = {a**3:.4f} 量级")
    json.dump({"rho": rho, "sigma": sigma, "a": a, "rows": rows},
              open(os.path.join(OUT, "logs", "B13_linear.json"), "w"), indent=1)


def c4_monotonicity(rho_grid=(0.0, 0.3, 0.7, 0.95), sig_grid=(0.15, 0.3, 0.6, 1.0, 2.0)):
    st_grid = np.linspace(0.0, np.pi, 49)
    rows = []
    worst = None
    for rho in rho_grid:
        for sigma in sig_grid:
            mu1 = np.array([1.0, 0.0], complex)
            mu2 = np.array([rho, np.sqrt(1 - rho ** 2)], complex)
            pes = [risk_euclid_numeric(mu1, mu2, st, sigma, n_grid=2048) for st in st_grid]
            diffs = np.diff(pes)
            a = np.sqrt(1 - rho) / sigma
            mono = bool(np.all(diffs >= -1e-9))
            rows.append({"rho": rho, "sigma": sigma, "a": a, "monotone": mono,
                         "min_diff": float(diffs.min())})
            if not mono and (worst is None or diffs.min() < worst["min_diff"]):
                worst = rows[-1]
            print(f"[C4] rho={rho} sigma={sigma} a={a:.2f}: monotone={mono} "
                  f"min_diff={diffs.min():.2e}", flush=True)
    n_mono = sum(1 for r in rows if r["monotone"])
    print(f"[C4] 单调: {n_mono}/{len(rows)}")
    json.dump({"rows": rows, "worst_violation": worst},
              open(os.path.join(OUT, "logs", "B13_monotonicity.json"), "w"), indent=1)


def c5_floor(K_grid=(1, 5, 20), st_grid=(0.5, 1.0, 2.0, 3.14), sigma=0.05,
             p=32, n_rep=4000, seed=0):
    """Prop6: 欧氏原型 vs oracle对齐原型的方向角方差。高SNR暴露相位地板。"""
    rng = np.random.RandomState(seed)
    mu = np.exp(1j * 0.3) * np.ones(p) / np.sqrt(p)
    rows = []
    for K in K_grid:
        for st in st_grid:
            e_eu, e_or = [], []
            for _ in range(n_rep):
                th = st * rng.randn(K)
                eps = (rng.randn(K, p) + 1j * rng.randn(K, p)) * (sigma / np.sqrt(2))
                z = np.exp(1j * th)[:, None] * mu[None, :] + eps
                mu_eu = z.mean(0)
                mu_or = (z * np.exp(-1j * th)[:, None]).mean(0)
                aerr = lambda m: float(np.sum(np.abs(m - mu) ** 2))
                e_eu.append(aerr(mu_eu)); e_or.append(aerr(mu_or))
            g1 = np.exp(-st ** 2 / 2)
            # 精确三项分解（Prop6修正版）:
            # MSE(欧氏) = (1-g1)^2 偏差(不随K缩小) + (1-g1^2)/K 方差地板 + pσ²/K 噪声
            mse_pred_eu = (1 - g1) ** 2 + (1 - g1 ** 2) / K + p * sigma ** 2 / K
            mse_pred_or = p * sigma ** 2 / K
            r = {"K": K, "sigma_th": st, "g1": g1,
                 "mse_euclid": float(np.mean(e_eu)), "mse_oracle_align": float(np.mean(e_or)),
                 "pred_euclid": mse_pred_eu, "pred_oracle": mse_pred_or,
                 "floor_share": (1 - g1 ** 2) / max(mse_pred_eu, 1e-12)}
            rows.append(r)
            print(f"[C5] K={K} st={st:.2f}: MSE_eu={r['mse_euclid']:.4f} "
                  f"(pred {mse_pred_eu:.4f}) MSE_or={r['mse_oracle_align']:.4f} "
                  f"(pred {mse_pred_or:.4f}) 地板占比={r['floor_share']:.2f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "logs", "B13_floor.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    c1_thm2()
    c2_crossing()
    c3_linear()
    c4_monotonicity()
    c5_floor()
    print("B13 DONE")
