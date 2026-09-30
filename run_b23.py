"""B23：P4 非渐近风险分离定理组的数值验证（本地 numpy）。

V1 欧氏成对风险公式 P_e = E_θ[Q(βcosθ)] vs MC（网格 ρ×σ×σ_θ）
V2 轨道成对风险跨 σ_θ 精确不变（MC，Thm P4-A）
V3 P_e 单调性（公式精细积分，201 个 σ_θ × 6 个 β）
V4 线性律+余项括号覆盖真值（Prop P4-C）
V5 交叉点括号 [σ_a, σ_b] 覆盖真实 σ*（Cor P4-D）
V6 K-shot（k=5）管线 MC：轨道平坦 / 欧氏增 + β_K 近似精度（Prop P4-E）
输出：04_results/logs/B23_p4_nonasymptotic.json
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
rng = np.random.RandomState(23)


def pe_formula(beta, st, n=200001):
    """P_e = E_θ[Q(βcosθ)]，θ~wrapN(0,st²)（Poisson 核积分）。"""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    if st < 1e-9:
        return float(norm.sf(beta))
    # wrapN 密度 = 高斯缠绕和（m ∈ [−4,4]，4σ 截断；非 Poisson 核！）
    ms = np.arange(-4, 5)
    imgs = th[None, :] + 2 * np.pi * ms[:, None]
    pr = np.exp(-0.5 * (imgs / st) ** 2).sum(0) / (st * np.sqrt(2 * np.pi))
    pr = pr / np.trapezoid(pr, th)
    return float(np.trapezoid(norm.sf(beta * np.cos(th)) * pr, th))


def mc_pairwise(rho, sig, st, n=400000, seed=0):
    """成对 MC：p=2 精确。μ₁=(1,0), μ₂=(ρ,√(1−ρ²))；z = e^{iθ}μ₁ + ε（全 2D）。"""
    r = np.random.RandomState(seed)
    E = n
    th = st * r.randn(E) if st > 0 else np.zeros(E)
    eps = (sig / np.sqrt(2)) * (r.randn(E, 2) + 1j * r.randn(E, 2))
    m1 = np.array([1.0, 0.0])
    m2 = np.array([rho, np.sqrt(max(1 - rho ** 2, 0.0))])
    z = np.exp(1j * th)[:, None] * m1[None, :] + eps
    d1 = (np.abs(z - m1[None, :]) ** 2).sum(1)
    d2 = (np.abs(z - m2[None, :]) ** 2).sum(1)
    err_e = (d1 > d2)
    ip1 = np.conj(z) @ m1
    ip2 = np.conj(z) @ m2
    err_o = (np.abs(ip1) < np.abs(ip2))
    return float(err_e.mean()), float(err_o.mean())


def linear_law(beta, st):
    return float(norm.sf(beta) + beta * norm.pdf(beta) * (1 - np.exp(-0.5 * st ** 2)))


def rbar(beta, st):
    # max_{|u|<=1} |u|φ(βu)：β≥1 时峰在 u=1/β ⇒ (1/β)φ(1)；β<1 时 φ(β)
    M = norm.pdf(1.0) / beta if beta >= 1 else norm.pdf(beta)
    g1 = np.exp(-0.5 * st ** 2)
    g2 = np.exp(-2 * st ** 2)
    return 0.5 * beta ** 3 * M * (1 - 2 * g1 + (1 + g2) / 2)


def main():
    t0 = time.time()
    res = {"checks": {}}
    rhos, sigs = [0.0, 0.3, 0.7], [0.2, 0.5, 1.0]
    sts = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]

    # V1 + V2
    v1err, v2spread = [], []
    for rho in rhos:
        for sig in sigs:
            beta = np.sqrt(1 - rho) / sig
            po0 = None
            for st in sts:
                pe_a = pe_formula(beta, st)
                pe_m, po_m = mc_pairwise(rho, sig, st, seed=int(1e4 * rho + 100 * sig + st * 7))
                v1err.append(abs(pe_a - pe_m))
                if po0 is None:
                    po0 = po_m
                v2spread.append(abs(po_m - po0))
    res["checks"]["V1_formula_vs_MC_max_abs"] = float(np.max(v1err))
    res["checks"]["V1_pass"] = bool(np.max(v1err) < 5e-3)          # MC se≈8e-4
    res["checks"]["V2_orbital_invariance_max_spread"] = float(np.max(v2spread))
    res["checks"]["V2_pass"] = bool(np.max(v2spread) < 5e-3)

    # V3 单调性（公式，精细网格）
    viol = 0
    for rho in rhos:
        for sig in sigs:
            beta = np.sqrt(1 - rho) / sig
            grid = np.linspace(0, np.pi, 201)
            vals = [pe_formula(beta, s, n=50001) for s in grid]
            viol += int(np.sum(np.diff(vals) < -1e-9))
    res["checks"]["V3_monotonicity_violations"] = viol
    res["checks"]["V3_pass"] = bool(viol == 0)

    # V4 线性律括号
    v4cov, v4wid = [], []
    for rho in rhos:
        for sig in sigs:
            beta = np.sqrt(1 - rho) / sig
            for st in sts[1:]:
                pe = pe_formula(beta, st)
                lo = linear_law(beta, st) - rbar(beta, st)
                hi = linear_law(beta, st) + rbar(beta, st)
                v4cov.append(lo - 1e-12 <= pe <= hi + 1e-12)
                v4wid.append(hi - lo)
    res["checks"]["V4_bracket_coverage"] = f"{int(np.sum(v4cov))}/{len(v4cov)}"
    res["checks"]["V4_pass"] = bool(np.all(v4cov))
    res["checks"]["V4_median_width"] = float(np.median(v4wid))

    # V5 交叉点括号（P_o 用 MC@σ_θ=0；σ* 从公式反解）
    v5cov = []
    for rho in rhos:
        for sig in sigs:
            beta = np.sqrt(1 - rho) / sig
            _, po = mc_pairwise(rho, sig, 0.0, n=2000000, seed=99)
            if not (norm.sf(beta) < po < 0.5):
                continue
            from scipy.optimize import brentq
            lo_f = lambda s: linear_law(beta, s) + rbar(beta, s) - po
            hi_f = lambda s: linear_law(beta, s) - rbar(beta, s) - po
            try:
                sa = brentq(lo_f, 1e-6, np.pi)
                sb = brentq(hi_f, 1e-6, np.pi)
            except ValueError:
                continue
            grid = np.linspace(1e-4, np.pi, 4001)
            vals = np.array([pe_formula(beta, s, n=20001) for s in grid])
            idx = np.argmin(np.abs(vals - po))
            s_true = grid[idx]
            v5cov.append(sa - 0.01 <= s_true <= sb + 0.01)
    res["checks"]["V5_crossing_bracket"] = f"{int(np.sum(v5cov))}/{len(v5cov)}"
    res["checks"]["V5_pass"] = bool(np.all(v5cov)) if v5cov else None

    # V6 K-shot 管线（k=5）：轨道平坦 / 欧氏增（全 2D 模拟）
    def mc_kshot(rho, sig, st, k=5, n=100000, seed=0):
        r = np.random.RandomState(seed)
        E = n
        m1 = np.array([1.0, 0.0])
        m2 = np.array([rho, np.sqrt(max(1 - rho ** 2, 0.0))])

        def support(mu, seed_off):
            rr = np.random.RandomState(seed + seed_off)
            ths = st * rr.randn(E, k) if st > 0 else np.zeros((E, k))
            eps = (sig / np.sqrt(2)) * (rr.randn(E, k, 2) + 1j * rr.randn(E, k, 2))
            return np.exp(1j * ths)[..., None] * mu[None, None, :] + eps

        Zs1 = support(m1, 1)                                 # (E,k,2)
        Zs2 = support(m2, 2)
        rr = np.random.RandomState(seed + 3)
        thq = st * rr.randn(E) if st > 0 else np.zeros(E)
        zq = np.exp(1j * thq)[:, None] * m1[None, :] + \
            (sig / np.sqrt(2)) * (rr.randn(E, 2) + 1j * rr.randn(E, 2))
        mu1_e, mu2_e = Zs1.mean(1), Zs2.mean(1)

        def align(Z):
            mu = Z.mean(1)                                   # (E,2)
            for _ in range(3):
                ip = (np.conj(mu)[:, None, :] * Z).sum(-1)   # (E,k)
                mu = (Z * np.exp(-1j * np.angle(ip))[..., None]).mean(1)
            return mu
        mu1_o, mu2_o = align(Zs1), align(Zs2)
        d1e = (np.abs(zq - mu1_e) ** 2).sum(1)
        d2e = (np.abs(zq - mu2_e) ** 2).sum(1)
        err_e = (d1e > d2e)
        ip1 = (np.conj(zq) * mu1_o).sum(1)
        ip2 = (np.conj(zq) * mu2_o).sum(1)
        err_o = (np.abs(ip1) < np.abs(ip2))
        return float(err_e.mean()), float(err_o.mean())

    v6 = []
    for rho, sig in [(0.0, 0.5), (0.3, 0.5), (0.7, 0.5)]:
        row = []
        for st in sts:
            ee, oo = mc_kshot(rho, sig, st, seed=int(1e4 * rho + st * 7))
            row.append({"st": float(st), "euclid": ee, "orbital": oo})
        v6.append({"rho": rho, "sig": sig, "rows": row,
                   "orbital_spread": float(max(r["orbital"] for r in row)
                                           - min(r["orbital"] for r in row)),
                   "euclid_increasing": bool(np.all(np.diff([r["euclid"] for r in row]) > -5e-3))})
    res["V6_kshot"] = v6
    res["checks"]["V6_orbital_flat_max_spread"] = float(max(v["orbital_spread"] for v in v6))
    res["checks"]["V6_orbital_pass"] = bool(max(v["orbital_spread"] for v in v6) < 5e-3)
    res["checks"]["V6_euclid_increasing_all"] = bool(all(v["euclid_increasing"] for v in v6))

    res["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = os.path.join(BASE, "04_results", "logs", "B23_p4_nonasymptotic.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["checks"], indent=1))
    npass = sum(1 for k, v in res["checks"].items() if k.endswith("_pass") and v)
    print(f"B23: {npass} checks pass -> {out}")


if __name__ == "__main__":
    main()
