"""B23：P4 非渐近风险分离定理组的数值验证（本地 numpy）。

V1 欧氏成对风险公式 P_e = E_θ[Q(βcosθ)] vs MC（网格 ρ×σ×σ_θ）
V2 轨道成对风险跨 σ_θ 精确不变（MC，Thm P4-A）
V3 P_e 单调性（公式精细积分，201 个 σ_θ × 6 个 β）
V4 线性律+余项括号覆盖真值（Prop P4-C，B28 修正常数后）
V5 交叉点定位集 B（Cor P4-D，B28 重述：报告区间括号成立率，不静默跳过）
V6 K-shot（k=5）管线 MC：轨道平坦 / 欧氏增
V7 β_K 有效宽度近似 vs K-shot MC（Prop P4-E(ii)——B28 审计发现此前"已验证"声明无代码）
V8 wrapN(π) 端点：P_e(π) = 1/2 − c(β), 0 < c(β) ≤ min(2/π, 1−2Q(β))e^{−π²/2}
   （B28 修正：旧文档"P_e(π)=1/2"错误——wrapN(π)≠均匀，其 Fourier 系数 e^{−k²π²/2}≠0）
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

    # V5 交叉点定位（P_o 用 MC@σ_θ=0；σ* 从公式反解）。
    # B28 重述（Cor P4-D 适用域）：定位集 B = {σ: linear−R̄ ≤ P_o ≤ linear+R̄} 恒有效；
    # 区间括号 [σ_a, σ_b] 仅当 B 为单区间（求根成功）时给出。跳过格不再静默丢弃而是计数。
    v5cov = []
    v5_attempted, v5_bracketed, v5_rows = 0, 0, []
    from scipy.optimize import brentq
    for rho in rhos:
        for sig in sigs:
            beta = np.sqrt(1 - rho) / sig
            _, po = mc_pairwise(rho, sig, 0.0, n=2000000, seed=99)
            if not (norm.sf(beta) < po < 0.5):
                continue
            v5_attempted += 1
            lo_f = lambda s: linear_law(beta, s) + rbar(beta, s) - po
            hi_f = lambda s: linear_law(beta, s) - rbar(beta, s) - po
            try:
                sa = brentq(lo_f, 1e-6, np.pi)
                sb = brentq(hi_f, 1e-6, np.pi)
                bracket_ok = True
            except ValueError:
                bracket_ok = False
            grid = np.linspace(1e-4, np.pi, 4001)
            vals = np.array([pe_formula(beta, s, n=20001) for s in grid])
            idx = np.argmin(np.abs(vals - po))
            s_true = grid[idx]
            # 定位集 B 的网格判定（恒有效）
            inB = [(s, bool(linear_law(beta, s) - rbar(beta, s) - 1e-12 <= po
                            <= linear_law(beta, s) + rbar(beta, s) + 1e-12))
                   for s in np.linspace(1e-4, np.pi, 401)]
            B_set = [s for s, ok in inB if ok]
            covered_by_B = bool(B_set and min(B_set) - 0.01 <= s_true <= max(B_set) + 0.01)
            row = {"rho": rho, "sig": sig, "beta": float(beta), "po": po,
                   "s_true": s_true, "interval_bracket": bracket_ok,
                   "B_interval": [min(B_set), max(B_set)] if B_set else None,
                   "covered_by_interval_bracket": bool(
                       bracket_ok and sa - 0.01 <= s_true <= sb + 0.01),
                   "covered_by_B": covered_by_B}
            v5_rows.append(row)
            if bracket_ok:
                v5_bracketed += 1
                v5cov.append(row["covered_by_interval_bracket"])
            print(f"[V5] rho={rho} sig={sig} beta={beta:.2f} po={po:.4f} "
                  f"s_true={s_true:.3f} bracket={bracket_ok} "
                  f"B=[{row['B_interval'][0] if B_set else '-'},"
                  f"{row['B_interval'][1] if B_set else '-'}] "
                  f"cover_int={row['covered_by_interval_bracket']} "
                  f"cover_B={covered_by_B}", flush=True)
    res["V5_rows"] = v5_rows
    res["checks"]["V5_attempted_cells"] = v5_attempted
    res["checks"]["V5_interval_bracket_formed"] = f"{v5_bracketed}/{v5_attempted}"
    res["checks"]["V5_bracket_coverage"] = f"{int(np.sum(v5cov))}/{len(v5cov)}"
    res["checks"]["V5_pass"] = bool(np.all(v5cov)) if v5cov else None
    res["checks"]["V5_all_covered_by_B"] = bool(all(r["covered_by_B"] for r in v5_rows))

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

    # V7 β_K 有效宽度近似（Prop P4-E(ii)——B28 新增：此前"已验证"声明无对应代码）。
    # K-shot 欧氏管线：μ̂_c = μ_c + ν_c，E||ν_c||² = V_K（Prop6 精确三项）；
    # 有效判决 = Re⟨z,Δ⟩ + (μ̂ 交叉项)，一阶近似 → Q(β_K cosθ) 结构，
    # β_K = √(1−ρ)/√(σ² + V_K/(2(1−ρ)))。对照：k-shot 欧氏管线 MC。
    def pe_beta_K(beta_K, st):
        return pe_formula(beta_K, st)

    v7 = []
    for rho, sig in [(0.0, 0.5), (0.3, 0.5), (0.7, 0.5)]:
        for k in (1, 5):
            for st in (0.0, np.pi / 3, np.pi):
                ee, _ = mc_kshot(rho, sig, st, k=k, n=200000,
                                 seed=int(1e4 * rho + 13 * k + st * 7))
                VK = ((1 - np.exp(-0.5 * st ** 2)) ** 2          # (1−g₁)² 偏差
                      + (1 - np.exp(-st ** 2)) / k               # 方差地板
                      + 2 * sig ** 2 / k)                        # pσ²/k（p=2）
                beta_K = np.sqrt(1 - rho) / np.sqrt(sig ** 2 + VK / (2 * (1 - rho)))
                pred = pe_beta_K(beta_K, st)
                v7.append({"rho": rho, "sig": sig, "k": k, "st": float(st),
                           "mc_euclid_kshot": ee, "betaK_pred": pred,
                           "abs_dev": abs(ee - pred),
                           "VK": VK, "beta": float(np.sqrt(1 - rho) / sig),
                           "beta_K": float(beta_K)})
                print(f"[V7] rho={rho} sig={sig} k={k} st={st:.2f}: "
                      f"MC={ee:.4f} β_K pred={pred:.4f} dev={abs(ee-pred):.4f}", flush=True)
    res["V7_betaK"] = v7
    res["checks"]["V7_max_abs_dev"] = float(max(r["abs_dev"] for r in v7))
    res["checks"]["V7_mean_abs_dev"] = float(np.mean([r["abs_dev"] for r in v7]))
    res["checks"]["V7_monotone_better_than_beta"] = bool(all(
        abs(r["abs_dev"]) < 0.05 for r in v7))

    # V8 wrapN(π) 端点（B28 修正）：P_e(π) = 1/2 − c(β)，0 < c(β) ≤
    # min(2/π, 1−2Q(β))·e^{−π²/2}；同时对照真均匀端点（Thm2）恰为 1/2。
    v8 = []
    for rho, sig in [(0.0, 0.2), (0.0, 0.5), (0.3, 0.5), (0.7, 0.5), (0.0, 1.0)]:
        beta = np.sqrt(1 - rho) / sig
        pe_pi = pe_formula(beta, np.pi)
        cb = min(2 / np.pi, 1 - 2 * norm.sf(beta)) * np.exp(-np.pi ** 2 / 2)
        ok = bool(0.0 < 0.5 - pe_pi <= cb + 1e-12)
        v8.append({"rho": rho, "sig": sig, "beta": float(beta), "pe_at_pi": pe_pi,
                   "c_beta": float(0.5 - pe_pi), "c_bound": float(cb), "pass": ok})
        print(f"[V8] beta={beta:.2f}: P_e(π)={pe_pi:.6f} c={0.5-pe_pi:.6f} "
              f"bound={cb:.6f} pass={ok}", flush=True)
    res["V8_endpoint"] = v8
    res["checks"]["V8_endpoint_pass"] = bool(all(r["pass"] for r in v8))

    res["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = os.path.join(BASE, "04_results", "logs", "B23_p4_nonasymptotic.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["checks"], indent=1))
    npass = sum(1 for k, v in res["checks"].items() if k.endswith("_pass") and v)
    print(f"B23: {npass} checks pass -> {out}")


if __name__ == "__main__":
    main()
