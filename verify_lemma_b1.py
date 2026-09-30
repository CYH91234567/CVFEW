"""Lemma B.1 数值证书（审计轮补齐）：wrapN(0,σ²) 密度 pr 在 (0,π) 严格递减。

原 theory_P4_nonasymptotic.md 声称"9×401 网格上 min pr' = −3.6e−8 < 0 无违例"，
但 B23 日志与 run_b23.py 均无对应字段/代码（第八轮审计发现项#3）。本脚本把该
证书补成可复现产物：解析导数 pr'(θ) = −Σ_m (θ+2πm)/σ² · φ_σ(θ+2πm)，
9 个 σ × 401 点网格，落盘 max pr'（最接近 0 的值，须 < 0）。

输出：04_results/logs/B23_lemmaB1_cert.json
"""
import json, os
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ST = np.linspace(np.pi / 9, np.pi, 9)           # 9 个 σ（含 π 端点，最难档）
TH = np.linspace(1e-4, np.pi - 1e-4, 401)       # (0,π) 开区间 401 点


def wrapn_pdf_and_deriv(th, st, m_half=25):
    imgs = th[..., None] + 2 * np.pi * np.arange(-m_half, m_half + 1)[None, ...]
    g = np.exp(-0.5 * (imgs / st) ** 2) / (st * np.sqrt(2 * np.pi))
    pdf = g.sum(-1)
    dphi = -(imgs / st ** 2) * g                # φ_σ 的导数
    return pdf, dphi.sum(-1)


def main():
    worst = {"max_pr_prime": -np.inf, "sigma": None, "theta": None}
    rows = []
    for st in ST:
        _, dpr = wrapn_pdf_and_deriv(TH, st)
        i = int(np.argmax(dpr))
        rows.append({"sigma": float(st), "max_dpr": float(dpr[i]),
                     "theta_at_max": float(TH[i])})
        if dpr[i] > worst["max_pr_prime"]:
            worst = {"max_pr_prime": float(dpr[i]), "sigma": float(st),
                     "theta_at_max": float(TH[i])}
    # 对称性证书 pr(u) > pr(π−u)，u∈(0,π/2) 开区间。
    # 数值口径：u=π/2 处恒等退化（π−u=u），网格最近点距 π/2 ~1e-16，
    # 差值低于机器精度 —— 排除端点半格距（δ=网格步长/2），如实注记。
    dth = float(TH[1] - TH[0])
    u_max = np.pi / 2 - dth / 2
    sym_rows = []
    for st in ST:
        pr, _ = wrapn_pdf_and_deriv(TH, st)
        u = TH[TH < u_max]
        # pr(π−u)：把 TH 换成 π−u
        pu_mirror, _ = wrapn_pdf_and_deriv(np.pi - u, st)
        sym_rows.append({"sigma": float(st),
                         "min_pr_u_minus_pr_piu": float(np.min(pr[TH < u_max] - pu_mirror)),
                         "u_max_excluded_endpoint": u_max})
    out = {"claim": "Lemma B.1: wrapN(0,σ²) 密度在 (0,π) 严格递减且 pr(u)>pr(π−u)",
           "grid": {"n_sigma": len(ST), "n_theta": len(TH),
                    "sigma_range": [float(ST[0]), float(ST[-1])],
                    "theta_range_open_interval": "(0,π)"},
           "max_pr_prime_over_grid": worst,
           "per_sigma_max_dpr": rows,
           "per_sigma_symmetry_min_diff": sym_rows,
           "verdict": {"strictly_decreasing": bool(worst["max_pr_prime"] < 0),
                       "symmetry_pr_u_gt_pr_piu": bool(all(r["min_pr_u_minus_pr_piu"] > 0
                                                          for r in sym_rows))}}
    path = os.path.join(BASE, "04_results", "logs", "B23_lemmaB1_cert.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out["verdict"], indent=1))
    print("max pr' over 9x401 grid:", worst)
    print("-> ", path)


if __name__ == "__main__":
    main()
