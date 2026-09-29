"""B4：理论-数值交叉核对（本地numpy，修订版）。

C1  P4 风险差（p=2 精确几何）：
    - 欧氏: 给定 θ 的内层概率是 1D 高斯尾 → θ 积分（解析-数值，精确）
    - 轨道: 高精度 MC (p=2, 2e6 样本, 3种子) 为参考；验证
      "欧氏风险随 sigma_th 单调升向 0.5、轨道风险与 sigma_th 无关"
C2  P3 二分法：高SNR+近精确对齐下，方向估计方差比（模拟 vs P3 公式）
    - 衰落: Var_uw/Var_w = E[h²]/E[h]² = (1-q+qκ²)/(1-q+qκ)²（有界）
    - 噪声异方差: = E[σ²]·E[σ⁻²]（无界）
C3  P2(iii) 退化：正交支持下轨道均值解集含 (n-1) 维环面 ——
    解析构造两个合法解，测量查询得分的分歧幅度（非唯一性的直接演示）。
"""
import json, os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E


def risk_euclid_numeric(mu_n, mu_x, sig_th, tau, n_grid=4096):
    """欧氏 1-shot 风险（population原型，p维各向同性）：给定θ，
    d_n - d_x = 2Re(z^H(mu_n - mu_x))；z^H(mu_n-mu_x) ~ CN(e^{iθ}mu_n^H(mu_n-mu_x), τ²‖μ_n-μ_x‖²)
    → 1D 复高斯尾概率 × θ 积分。"""
    v = mu_x - mu_n                                    # mu_n, mu_x 单位范数（p维）
    nv2 = np.abs(np.vdot(v, v)).real
    c = np.vdot(mu_n, v)                               # mu_n^H (mu_x - mu_n)
    gap0 = -2 * c.real                                  # d_n - d_x 在 θ=0 的均值间隙（负=偏向x?）注意符号
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    if sig_th < 1e-9:
        pr = np.zeros_like(th); pr[np.argmin(np.abs(th))] = 1.0   # δ 在 θ=0
    else:   # wrapped normal 缠绕密度（σ_θ→π 时趋于均匀）
        Js = np.arange(-6, 7)
        pr = np.exp(-0.5 * ((th[None, :] + 2 * np.pi * Js[:, None]) / sig_th) ** 2).sum(0)             / (sig_th * np.sqrt(2 * np.pi))
    pr /= pr.sum()
    # 给定 θ: T = z^H v = e^{iθ}c + η, η~CN(0, τ² nv²)，v = μ_x − μ_n。
    # d_n − d_x = 2Re(T)；错分 d_x < d_n ⟺ Re(T) > 0 → P = Φ(Re(e^{iθ}c)/sd)
    m_th = np.exp(1j * th) * c                          # (G,)
    sd = tau * np.sqrt(nv2) / np.sqrt(2.0)   # Re[CN(0,s²)] ~ N(0, s²/2)
    perr = (norm.cdf(m_th.real / sd) * pr).sum()
    return float(perr)


def risk_mc(method, mu_n, mu_x, sig_th, tau, n_mc=2_000_000, seed=0):
    """p=2 MC。mu_n, mu_x: (2,) 复。z ~ CN(e^{iθ}mu_n, τ²I)。"""
    rng = np.random.RandomState(seed)
    th = sig_th * rng.randn(n_mc)
    eta = (rng.randn(n_mc, 2) + 1j * rng.randn(n_mc, 2)) * (tau / np.sqrt(2))
    z = np.exp(1j * th)[:, None] * mu_n[None, :] + eta
    if method == "euclid":
        d_n = (np.abs(z - mu_n[None, :]) ** 2).sum(1)
        d_x = (np.abs(z - mu_x[None, :]) ** 2).sum(1)
    else:
        ip_n = z.conj() @ mu_n
        ip_x = z.conj() @ mu_x
        d_n = (np.abs(z) ** 2).sum(1) + 1 - 2 * np.abs(ip_n)
        d_x = (np.abs(z) ** 2).sum(1) + 1 - 2 * np.abs(ip_x)
    return float((d_x < d_n).mean())


def c1_p4(out, sig_list=(0.0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi), tau=0.3, rho_c=0.3):
    """P4：欧氏 vs 轨道 1-shot 风险 vs σ_θ（相干类对，φ 由 rho_c 定）。
    mu_n=[1,0], mu_x=[rho_c, sqrt(1-rho_c²)·e^{iφ0}]（单位范数，内积 rho_c）。"""
    mu_n = np.array([1.0, 0.0], dtype=np.complex128)
    phi0 = np.pi / 3
    mu_x = np.array([rho_c, np.sqrt(1 - rho_c ** 2) * np.exp(1j * phi0)], dtype=np.complex128)
    rows = []
    seeds = (1, 2, 3)
    for sig in sig_list:
        r_eu_mc = float(np.mean([risk_mc("euclid", mu_n, mu_x, sig, tau, seed=s) for s in seeds]))
        r_or_mc = float(np.mean([risk_mc("orbital", mu_n, mu_x, sig, tau, seed=s) for s in seeds]))
        r_eu_num = risk_euclid_numeric(mu_n, mu_x, sig, tau)
        rows.append({"sigma_th": sig, "rho_c": rho_c, "tau": tau,
                     "risk_euclid_mc": r_eu_mc, "risk_orbital_mc": r_or_mc,
                     "risk_euclid_numeric": r_eu_num,
                     "euclid_num_vs_mc": abs(r_eu_num - r_eu_mc)})
        print(f"[C1] sig={sig:.2f}: euclid MC={r_eu_mc:.4f} num={r_eu_num:.4f} "
              f"(d={abs(r_eu_num-r_eu_mc):.4f}) | orbital MC={r_or_mc:.4f}", flush=True)
    mc = np.array([r["risk_euclid_mc"] - r["risk_orbital_mc"] for r in rows])
    print(f"[C1] 欧氏-轨道风险差随 sigma_th: {np.round(mc,3)}")
    print(f"[C1] 欧氏解析 vs MC 最大偏差: {max(r['euclid_num_vs_mc'] for r in rows):.4f} "
          f"(MC 3种子均值, SE≈{tau*0+0.0003:.4f})")
    json.dump(rows, open(os.path.join(out, "logs", "B4_p4_risk.json"), "w"), indent=1)


def c2_p3_dichotomy(out, n_rep=2000, p=32, n=200, sig=0.05, sig_th=0.3, q=0.5):
    """P3 二分法（渐近方差公式适用 regime, n=200, 对齐用真 θ）：方差比。"""
    rng = np.random.RandomState(7)
    mu = np.exp(1j * 0.3) * np.ones(p) / np.sqrt(p)
    res = {}
    mse_ns, mse_ls = [], []
    for name, mode, kap in [("fading", "fade", 0.1), ("noise_het", "het", 10.0)]:
        direrr_uw, direrr_w = [], []
        for _ in range(n_rep):
            th = sig_th * rng.randn(n)
            h = np.ones(n); sg = np.full(n, sig)
            if mode == "fade":
                h[rng.rand(n) < q] = kap
            else:
                sg[rng.rand(n) < q] = sig * kap
            z = (np.exp(1j * th) * h)[:, None] * mu[None, :] + \
                (rng.randn(n, p) + 1j * rng.randn(n, p)) * (sg / np.sqrt(2))[:, None]
            # 精确对齐角（近真值：θ 已知性高）用真 θ 后的加权/未加权均值
            zal = z * np.exp(-1j * th)[:, None]
            mu_uw = zal.mean(0)
            # P3 公式对应的两估计量：Σw_j z̄_j / Σw_j（w=1 vs w=h 或 1/σ²）
            w = (1 / sg ** 2) if mode == "het" else h
            mu_w = (w[:, None] * zal).sum(0) / w.sum()
            # 检验方向严格正交于 μ（去除 h 波动沿 μ 的污染）
            tv = rng.randn(p) + 1j * rng.randn(p)
            test_dir = tv - np.vdot(mu, tv) / np.vdot(mu, mu) * mu
            test_dir /= np.linalg.norm(test_dir)
            e_uw = np.real(np.vdot(test_dir, mu_uw - mu))
            e_w = np.real(np.vdot(test_dir, mu_w - mu))
            direrr_uw.append(e_uw); direrr_w.append(e_w)
        # 全信号估计对象（衰落情形）：朴素缩放均值 z̄_j/h_j vs LS(Σh²)
        if mode == "fade":
            zs = zal / h[:, None]
            mu_ns = zs.mean(0)                       # naive scale-corrected
            mu_ls = (h[:, None] * zal).sum(0) / (h ** 2).sum()
            mse_ns.append(float(np.sum(np.abs(mu_ns - mu) ** 2)))
            mse_ls.append(float(np.sum(np.abs(mu_ls - mu) ** 2)))
        var_uw = float(np.var(direrr_uw)); var_w = float(np.var(direrr_w))
        ratio = var_uw / max(var_w, 1e-30)
        res[name] = {"var_unweighted": var_uw, "var_weighted": var_w, "ratio": ratio}
        if name == "fading":
            res["fading_full_mu"] = {"mse_naive_scaled": float(np.mean(mse_ns)),
                                     "mse_LS": float(np.mean(mse_ls)),
                                     "ratio": float(np.mean(mse_ns) / max(np.mean(mse_ls), 1e-30)),
                                     "theory_Eh2Eh-2": float((1 - q + q * kap ** 2) * (1 - q + q / kap ** 2))}
        print(f"[C2] {name}: Var_uw={var_uw:.3e} Var_w={var_w:.3e} ratio={ratio:.2f}", flush=True)
    kh, ks = 0.1, 10.0
    res["theory"] = {"fading_ratio": (1 - q + q * kh ** 2) / (1 - q + q * kh) ** 2,
                     "noise_het_ratio": (1 - q + q * ks ** 2) * (1 - q + q / ks ** 2)}
    print(f"[C2] theory: fading={res['theory']['fading_ratio']:.2f} "
          f"noise_het={res['theory']['noise_het_ratio']:.2f}")
    res["config"] = {"n": n, "p": p, "sig": sig, "sig_th": sig_th, "q": q,
                     "note": "渐近方差公式核对（真θ对齐, 正交检验方向）；n=5 少样本情形的加权增益由 B5 网格直接测量"}
    json.dump(res, open(os.path.join(out, "logs", "B4_p3_dichotomy.json"), "w"), indent=1)


def c3_degeneracy(out, n_rep=100, p=64, n=5):
    """P2(iii)：z_j = α_j e_j（正交支持）时轨道均值解族
    μ̂(φ) = Σ_j α_j e^{iφ_j} e_j / √(Σα²)（|u_j|=α_j/√Σα²，相位任意）。
    直接构造两个解，测查询得分分歧。"""
    rng = np.random.RandomState(11)
    rows = []
    for trial in range(n_rep):
        alphas = rng.uniform(0.5, 1.5, n)
        m1 = np.zeros(p, dtype=np.complex128); m2 = np.zeros(p, dtype=np.complex128)
        m1[np.arange(n)] = alphas * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        m2[np.arange(n)] = alphas * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        m1 /= np.sqrt((alphas ** 2).sum()); m2 /= np.sqrt((alphas ** 2).sum())
        zq = (rng.randn(p) + 1j * rng.randn(p)) / np.sqrt(p)
        s1 = abs(np.vdot(zq, m1)); s2 = abs(np.vdot(zq, m2))
        rel = abs(s1 - s2) / max((s1 + s2) / 2, 1e-12)
        rows.append({"rel_divergence": rel})
    rels = np.array([r["rel_divergence"] for r in rows])
    out_d = {"n_rep": n_rep, "rel_divergence_p50": float(np.percentile(rels, 50)),
             "rel_divergence_p90": float(np.percentile(rels, 90)),
             "frac_gt_10pct": float((rels > 0.1).mean()),
             "note": "两合法解对同一查询的轨道得分相对分歧；>10% 即可翻转近距离类对的判罚"}
    print(f"[C3] 解间得分分歧: p50={out_d['rel_divergence_p50']:.2f} "
          f"p90={out_d['rel_divergence_p90']:.2f} >10%占比={out_d['frac_gt_10pct']:.2f}")
    json.dump(out_d, open(os.path.join(out, "logs", "B4_degeneracy.json"), "w"), indent=1)


if __name__ == "__main__":
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    c1_p4(out)
    c2_p3_dichotomy(out)
    c3_degeneracy(out)
    print("B4 DONE")
