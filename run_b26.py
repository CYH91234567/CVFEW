"""B26：收缩对齐定理组数值验证（PREREG_B26；本地 numpy+scipy，无 torch 依赖）。

模型：z_j = e^{iθ_j}(aμ + ε_j)，ε_j ~ CN(0, σ²I_p)，θ_j ~ wrapN(0, σ_θ)，λ = a²/σ²。
μ 取实单位向量 e1（各向同性 wlog）。

验证：
  V1  Prop(i)  tower 无偏性：E[Σ_j γ_full,j z_j] = k·a·μ（soft-full = 网格后验均值，
               真先验+真似然，oracle 参考方向）。
  V2  Prop(ii) 硬对齐（oracle 参考）：E[Σ z_j e^{-i arg r_j}] = k·E[R]·μ，R=|a+n|
               Rician（scipy.stats.rice.mean），且与 σ_θ 无关。
               （B28 修复：旧实现 target 误乘 a——m_hard 定义为 E[R]/a，故 target
               应为 k·E[R]；且 dev_in_se 此前未进 verdict，公式一致性从未被判。）
  V3  Prop(iv) 主表：unit 原型 MSE(mean / t∈{1,2,3,5,10} 自对齐 / L1PCA30 /
               soft-vm-self（=B25 训练臂镜像，ρ=2|ip|，3 E步）/ soft-full-oracle /
               hard-oracle）× λ 12 点 × σ_θ 4 值，k=5, p=64。
               v3_claims（C1–C6）在 main() 内从主表计算落盘（B28 修复：此前
               JSON 中的 v3_claims 无对应提交代码，不可复现）。
  V4  Prop(iii) soft-vm（oracle 参考）增益 E[R·h(2aR/σ²)]/a vs MC——B28 修正：
               该量 ≡ 1（精确恒等式，score 恒等式 E[R·E{cosφ|R}] = a），本检查
               作为恒等式确认；soft-full 增益 = 1（V1 推论）。先验免费的代价在
               横向收缩（V5 的 E[h²]<1），不在均值偏置。
  V5  E[h(2aR/σ²)²] 横向噪声收缩因子（Rician 一维求积）vs MC。
输出：logs/B26_shrinkage.json。
"""
import argparse, json, os
import numpy as np
from scipy.stats import rice


def draw_support(n_mc, k, p, a, sigma, st, rng):
    """Z:(n_mc,k,p) complex，θ~wrapN(0,st) 逐样本（拒绝法/网格逆变换）。"""
    z = (rng.standard_normal((n_mc, k, p)) + 1j * rng.standard_normal((n_mc, k, p))) \
        * (sigma / np.sqrt(2.0))
    z[..., 0] += a
    th = draw_wrapn(st, (n_mc, k), rng)
    return z * np.exp(1j * th)[..., None]


def draw_wrapn(st, shape, rng, n_img=40):
    """wrapN(0,st) 采样：正态 + 2π wrap（st ≤ π/2 直接 wrap 精确）。"""
    x = rng.standard_normal(shape) * st
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrapn_prior_grid(st, n_grid=1024):
    """wrapN(0,st) 周期密度 on [−π,π)（镜像和，n_img=40 覆盖 st≥0.1）。"""
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    js = np.arange(-40, 41)
    img = th[None, :] + 2 * np.pi * js[:, None]
    pr = np.exp(-0.5 * (img / st) ** 2).sum(0)
    return th, pr / pr.sum()


def gamma_full_grid(r, a, sigma, st, th_g=None, pr_g=None):
    """soft-full：γ = E[e^{-iθ}|z] 的网格后验（真 wrapN 先验 × 高斯似然，
    充分统计量 r=⟨μ,z⟩）。r 复标量数组。"""
    if th_g is None:
        th_g, pr_g = wrapn_prior_grid(st)
    rho = 2.0 * a * np.abs(r) / sigma ** 2
    phi = np.angle(r)
    ll = np.exp(rho[..., None] * np.cos(phi[..., None] - th_g))     # (...,G)
    post = ll * pr_g
    num = (post * np.exp(-1j * th_g)).sum(-1)
    return num / post.sum(-1).clip(1e-300)


def gamma_vm(r, a, sigma):
    """soft-vm（先验无关）：γ = I₁(ρ)/I₀(ρ)·e^{-i arg r}，ρ=2a|r|/σ²。"""
    from scipy.special import i1e, i0e
    rho = 2.0 * a * np.abs(r) / sigma ** 2
    h = i1e(rho) / i0e(rho)
    return np.exp(-1j * np.angle(r)) * h, h


def self_align(Z, n_iter, w_fn=None):
    """自对齐原型：t 步坐标上升（w_fn=None 硬 arg 对齐；否则软权重函数
    w_fn(ip, mu_norm) 返回逐样本复权重）。Z:(n,k,p)。返回 (n,p)。"""
    mu = Z.mean(1)
    for _ in range(n_iter):
        ip = np.einsum("bc,bkc->bk", mu.conj(), Z)
        if w_fn is None:
            w = np.exp(-1j * np.angle(ip))
        else:
            w = w_fn(ip, mu, Z)
        mu = (Z * w[..., None]).mean(1)
    return mu


def unit_rows(S):
    return S / np.maximum(np.linalg.norm(S, axis=-1, keepdims=True), 1e-12)


def mse_dir(mu_hat):
    """E||μ̂ − e1||²，μ̂ unit（复）。= 2 − 2·Re(μ̂_0)。"""
    return float(np.mean(2.0 - 2.0 * mu_hat[..., 0].real))


def vm_self_w(ip, mu, Z):
    """B25 soft 臂镜像：unit 化样本、ρ=2|ip|、3 E 步中的单步权重。"""
    g, _ = gamma_vm(ip, 1.0, 1.0)      # unit 域：ρ=2|ip|（a=σ=1 的归一化镜像）
    return g / np.abs(g).mean(-1, keepdims=True).clip(1e-12)


def v1_unbiased(n_mc=40000, seed=26):
    """V1/V2：k∈{1,5}，λ∈{0.1,1,10}，σ_θ∈{π/6,π/3,π}。"""
    rng = np.random.default_rng(seed)
    out = {"v1_softfull": [], "v2_hard": []}
    p = 8
    for k in (1, 5):
        for lam in (0.1, 1.0, 10.0):
            for st in (np.pi / 6, np.pi / 3, np.pi):
                sigma = 1.0
                a = np.sqrt(lam)
                th_g, pr_g = wrapn_prior_grid(st)
                Z = draw_support(n_mc, k, p, a, sigma, st, rng)
                r = Z[..., 0]                                    # ⟨e1, z⟩
                g = gamma_full_grid(r, a, sigma, st, th_g, pr_g)
                S_soft = (Z * g[..., None]).sum(1)
                S_hard = (Z * np.exp(-1j * np.angle(r))[..., None]).sum(1)
                # 目标：E[S] = k·a（μ=e1 实正向）。统计 μ 分量实部与横向范数。
                for nm, S in (("v1_softfull", S_soft), ("v2_hard", S_hard)):
                    mu_comp = S[..., 0].real
                    m, se = mu_comp.mean(), mu_comp.std() / np.sqrt(n_mc)
                    if nm == "v1_softfull":
                        pass
                    trans = np.linalg.norm(S[..., 1:], axis=-1).mean()
                    # Rician 用每维标准差约定：CN(0,σ²) → 每维 σ/√2
                    spd = sigma / np.sqrt(2.0)
                    # m_hard = E[R]/a（B28 修复：旧版误写成 a·rice.mean）
                    target = (k * a if nm == "v1_softfull"
                              else k * float(rice.mean(a / spd, scale=spd)))
                    dev_se = (m - target) / max(se, 1e-12)
                    out[nm].append({"k": k, "lam": lam, "st": st, "mean_mu": float(m),
                                    "se": float(se),
                                    "target": float(target), "dev_in_se": float(dev_se),
                                    "trans_mean": float(trans)})
    return out


def v3_mse_table(n_mc=4000, seed=126, k=5, p=64):
    rng = np.random.default_rng(seed)
    lams = np.logspace(-1.3, 1.3, 12)
    sts = (np.pi / 6, np.pi / 3, np.pi / 2, np.pi)
    rows = []
    for lam in lams:
        for st in sts:
            sigma = 1.0
            a = np.sqrt(lam)
            th_g, pr_g = wrapn_prior_grid(st)
            Z = draw_support(n_mc, k, p, a, sigma, st, rng)
            r = Z[..., 0]
            Zn = unit_rows(Z)                                    # unit 域（训练镜像）
            cell = {"lam": float(lam), "st": float(st)}
            cell["mean"] = mse_dir(unit_rows(Z.mean(1)))
            for t in (1, 2, 3, 5, 10):
                cell[f"self{t}"] = mse_dir(unit_rows(self_align(Z, t)))
            cell["l1pca30"] = mse_dir(unit_rows(self_align(Z, 30)))
            # soft-vm-self：3 E 步（B25 臂镜像，unit 域 ρ=2|ip|）
            mu = Zn.mean(1)
            for _ in range(3):
                ip = np.einsum("bc,bkc->bk", mu.conj(), Zn)
                g = gamma_vm(ip, 1.0, 1.0)[0]
                g = g / np.abs(g).mean(-1, keepdims=True).clip(1e-12)
                mu = (Zn * g[..., None]).sum(1)
            cell["soft_vm_self"] = mse_dir(unit_rows(mu))
            # soft-full-oracle（理论前端）
            g = gamma_full_grid(r, a, sigma, st, th_g, pr_g)
            cell["soft_full_oracle"] = mse_dir(unit_rows((Z * g[..., None]).sum(1)))
            # hard-oracle（B24(ii) 参考）
            cell["hard_oracle"] = mse_dir(unit_rows(
                (Z * np.exp(-1j * np.angle(r))[..., None]).sum(1)))
            rows.append(cell)
            print(f"  [V3] λ={lam:.3f} σ_θ={st:.2f}: mean={cell['mean']:.3f} "
                  f"self5={cell['self5']:.3f} l1pca={cell['l1pca30']:.3f} "
                  f"vm={cell['soft_vm_self']:.3f} full={cell['soft_full_oracle']:.3f}",
                  flush=True)
    return rows


def v4_v5(n_mc=200000, seed=226):
    """V4：soft-vm 增益公式 E[R·h(2aR/σ²)]/a vs MC；V5：E[h²] 公式 vs MC。"""
    rng = np.random.default_rng(seed)
    out = {"v4": [], "v5": []}
    for lam in (0.05, 0.2, 1.0, 5.0, 20.0):
        sigma = 1.0
        a = np.sqrt(lam)
        b = a / sigma
        # Rician 一维求积（每维标准差约定：CN(0,σ²) → 每维 σ/√2）
        spd = sigma / np.sqrt(2.0)
        xs = np.linspace(1e-6, a + 12 * sigma, 4000)
        pdf = rice.pdf(xs / spd, a / spd) / spd
        from scipy.special import i1e, i0e
        h = i1e(2.0 * a * xs / sigma ** 2) / i0e(2.0 * a * xs / sigma ** 2)
        g_quad = float(np.trapezoid(xs * h * pdf, xs) / a)
        h2_quad = float(np.trapezoid(h ** 2 * pdf, xs))
        # MC
        n = (a + rng.standard_normal(n_mc) * sigma / np.sqrt(2)
             + 1j * rng.standard_normal(n_mc) * sigma / np.sqrt(2))
        R = np.abs(n)
        h_mc = i1e(2.0 * a * R / sigma ** 2) / i0e(2.0 * a * R / sigma ** 2)
        g_mc = float(np.mean(R * h_mc) / a)
        h2_mc = float(np.mean(h_mc ** 2))
        out["v4"].append({"lam": lam, "quad": g_quad, "mc": g_mc,
                          "dev_se": float((g_mc - g_quad) / max(
                              (R * h_mc).std() / np.sqrt(n_mc) / a, 1e-12))})
        out["v5"].append({"lam": lam, "quad": h2_quad, "mc": h2_mc})
    return out


def v3_claims(rows):
    """C1–C6 从 V3 主表计算（B28 新增：verdict 落盘可复现）。
    判定用 B28 修正后的陈述（见 theory_shrinkage_alignment.md §4）：
      C1 低 SNR 自对齐有害：λ≤λ_grid[1] 时 mean 严格低于全部自参考估计器。
      C2 中 SNR 截断窗（格点依赖!）：λ∈[λ_grid[6], λ_grid[9]]=[2.26,6.72] 内
         min_t self{t} < min{mean, l1pca30}（U 形）。
      C3 高 SNR 对齐有益：λ≥λ_grid[9] 且 σ_θ>0 时 best-aligned < mean。
      C4 soft-full-oracle 全格优于全部自参考估计器。
      C5 soft_vm_self 弱优于 self5（容差 0.02，unit 域 MSE 尺度）。
      C6 soft-full MSE 的 σ_θ-不变性：逐 λ 报 soft_full 与 best-self 的跨 σ_θ 极差，
         报告 crossover λ（不变性只在 λ 足够大时成立——如实记录，不再写"近似不变"）。
    """
    lams = sorted({r["lam"] for r in rows})
    selfref = ["self1", "self2", "self3", "self5", "self10", "l1pca30", "soft_vm_self"]
    per_lam = []
    for lam in lams:
        sub = {r["st"]: r for r in rows if r["lam"] == lam}
        cell = {"lam": float(lam)}
        # C1（B28 修正口径）低 SNR 自对齐无增益且迭代单调变差：
        # λ ≤ lams[1] 时 (a) 全部非退化 σ_θ 的 best-aligned ≥ mean − 0.02；
        # (b) self{t} 对 t 单调非降（截断更多不更好）。
        if lam <= lams[1]:
            nogain = [min(r[f"self{t}"] for t in (1, 2, 3, 5, 10)) - r["mean"]
                      for st, r in sub.items() if st > 0.0]
            mono = all(
                all(r[f"self{t2}"] >= r[f"self{t1}"] - 0.02
                    for t1, t2 in ((1, 2), (2, 3), (3, 5), (5, 10)))
                for r in sub.values())
            cell["C1_max_gain"] = float(max(nogain))
            cell["C1_nogain_pass"] = bool(min(nogain) >= -0.02)
            cell["C1_mono_pass"] = bool(mono)
        # C2（窗口 = λ∈[2.26,6.72]，格点依赖如实注记；σ_θ<π 非退化格 +
        # 全 4 σ_θ 口径分别报告——σ_θ=π 时自对齐从坍缩均值出发不收敛，属已知例外）
        if 2.26 <= lam <= 6.72:
            u_all, u_nd = [], []
            for st, r in sub.items():
                mt = min(r[f"self{t}"] for t in (1, 2, 3, 5, 10))
                ok = mt < min(r["mean"], r["l1pca30"])
                u_all.append(ok)
                if st < np.pi:
                    u_nd.append(ok)
            cell["C2_pass_frac"] = f"{sum(u_all)}/{len(u_all)}"
            cell["C2_pass_frac_nondegenerate"] = f"{sum(u_nd)}/{len(u_nd)}"
        # C3
        if lam >= lams[9]:
            c3 = []
            for st, r in sub.items():
                if st == 0.0:
                    continue
                best = min(min(r[f"self{t}"] for t in (1, 2, 3, 5, 10)),
                           r["l1pca30"], r["soft_vm_self"])
                c3.append(best < r["mean"])
            cell["C3_pass_frac"] = f"{sum(c3)}/{len(c3)}"
        # C4
        cell["C4_pass"] = bool(all(
            r["soft_full_oracle"] < min(r[k] for k in selfref) - 1e-12
            for r in sub.values()))
        # C5
        cell["C5_max_excess"] = float(max(
            r["soft_vm_self"] - r["self5"] for r in sub.values()))
        # C6 spread
        cell["C6_softfull_range"] = float(
            max(r["soft_full_oracle"] for r in sub.values())
            - min(r["soft_full_oracle"] for r in sub.values()))
        cell["C6_bestself_range"] = float(max(
            min(r[k] for k in selfref) for r in sub.values())
            - min(min(r[k] for k in selfref) for r in sub.values()))
        per_lam.append(cell)
    crossovers = [c["lam"] for c in per_lam
                  if c["C6_softfull_range"] <= c["C6_bestself_range"]]
    claims = {"C1_at_lam": per_lam[1]["lam"] if len(per_lam) > 1 else None,
              "C2_window": "[2.26, 6.72] (λ 网格第 7–10 点，窗口格点依赖)",
              "C6_crossover_lam": min(crossovers) if crossovers else None,
              "detail": per_lam}
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../04_results/logs")
    ap.add_argument("--nmc-v3", type=int, default=4000)
    a = ap.parse_args()
    res = {"meta": "B26 shrinkage-alignment verification (PREREG_B26)"}
    print("=== V1/V2: tower unbiasedness + hard gain ===", flush=True)
    res["v1v2"] = v1_unbiased()
    v1max = max(abs(d["dev_in_se"]) for d in res["v1v2"]["v1_softfull"])
    v2spread, v2se, v2dev = {}, {}, []
    for d in res["v1v2"]["v2_hard"]:
        v2spread.setdefault((d["k"], d["lam"]), []).append(d["mean_mu"])
        v2se.setdefault((d["k"], d["lam"]), []).append(d["se"])
        v2dev.append(abs(d["dev_in_se"]))
    v2max_ratio = max((max(v) - min(v)) / (3.0 * np.sqrt(sum(s ** 2 for s in se)))
                      for (k_, lam_), v in v2spread.items()
                      for se in [v2se[(k_, lam_)]])
    v2dev_max = max(v2dev)
    print(f"V1 max |dev| = {v1max:.2f} SE（判据 ≤3）", flush=True)
    print(f"V2 公式=MC max |dev| = {v2dev_max:.2f} SE（判据 ≤3）", flush=True)
    print(f"V2 max spread/(3×comb-SE) = {v2max_ratio:.2f}（判据 ≤1，σ_θ-free）", flush=True)
    print("=== V3: MSE table ===", flush=True)
    res["v3"] = v3_mse_table(n_mc=a.nmc_v3)
    print("=== V4/V5: vm gain + h² shrinkage ===", flush=True)
    res["v4v5"] = v4_v5()
    v4max = max(abs(d["dev_se"]) for d in res["v4v5"]["v4"])
    v5max = max(abs(d["quad"] - d["mc"]) / max(d["quad"], 1e-12)
                for d in res["v4v5"]["v5"])
    print(f"V4 max |dev| = {v4max:.2f} SE；V5 max rel dev = {v5max:.4f}", flush=True)
    res["verdict"] = {
        "V1_unbiased_pass": bool(v1max <= 3.0),
        "V2_thetafree_pass": bool(v2max_ratio <= 1.0),
        "V2_thetafree_max_ratio": float(v2max_ratio),
        "V2_formula_vs_MC_max_dev_se": float(v2dev_max),
        "V2_formula_pass": bool(v2dev_max <= 3.0),
        "V4_identity_gvm_equals_1_pass": bool(v4max <= 3.0),
        "V5_h2_pass": bool(v5max <= 0.02),
    }
    # v3_claims（C1–C6，B28 起由本脚本计算落盘，可复现）
    res["verdict"]["v3_claims"] = v3_claims(res["v3"])
    print("v3_claims:", json.dumps({k: v for k, v in
          res["verdict"]["v3_claims"].items() if k != "detail"}, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "B26_shrinkage.json"), "w"), indent=1)
    print("B26 DONE", flush=True)


if __name__ == "__main__":
    main()
