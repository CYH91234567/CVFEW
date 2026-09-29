"""B18 分析：规范化（选截面）vs 免截面（轨道/PhaseMAP）的割迹代价。

用法: python analyze_b18.py [--synth 04_results/logs/B18_synth.json] [--real ...]
输出: 04_results/tables/T8_b18_synth.md / T9_b18_real.md, figures/F8_b18_*.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ORDER = ["euclid", "canon_first", "canon_mean", "canon_max", "canon_wmean", "canonX_max",
         "canon_ref", "canonX_ref", "orbital", "phasemap_ml", "phasemap_ok"]


def fmt(v):
    return "—" if v is None else f"{v:.1f}"


def synth_report(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]
    ps = sorted({r["p"] for r in recs})
    rs = sorted({r["ratio"] for r in recs})
    sts = sorted({round(r["sigma_th"], 3) for r in recs})
    grid = {(r["p"], r["ratio"], round(r["sigma_th"], 3)): r["acc"] for r in recs}
    L = []
    L.append("# T8：B18 合成网格 —— 规范化（截面）vs 免截面（轨道/PhaseMAP）\n")
    L.append(f"> K={recs[0]['K']}, k={recs[0]['k']}, m={recs[0]['m']}, ρ={recs[0]['rho']}，"
             f"每格 {recs[0]['n_ep']} episode，chance={100/recs[0]['K']:.0f}%。"
             f"噪声按能量比 r 参数化（σ=√(r/p)，跨 p 可比）。PP 单位。\n")

    for p in ps:
        L.append(f"\n## p={p}\n")
        L.append("| r | " + " | ".join(f"σ_θ={s:.2f}" for s in sts) + " |")
        L.append("|" + "---|" * (len(sts) + 1))
        for r in rs:
            # 每格取最强规范化 vs 免截面
            line = [f"**r={r:g}**"]
            for s in sts:
                a = grid[(p, r, s)]
                line.append(f"orb {100*a['orbital']:.1f} / ml {100*a['phasemap_ml']:.1f} / "
                            f"cxR {100*a['canonX_ref']:.1f} / cR {100*a['canon_ref']:.1f} / "
                            f"eucl {100*a['euclid']:.1f}")
            L.append("| " + " | ".join(line) + " |")

    L.append("\n## 关键对照（PP，按 (p, r) 汇总；σ_θ≥π/3 为压力格）\n")
    L.append("| p | r | 最强规范化 canonX_ref − orbital | canon_ref − orbital | "
             "canon_max − canon_ref（选择偏倚） | phasemap_ml − max(euclid,orbital) |")
    L.append("|---|---|---|---|---|---|")
    stress = [s for s in sts if s >= np.pi / 3 - 1e-6]
    rows = []
    for p in ps:
        for r in rs:
            def m(key, ss):
                return np.mean([grid[(p, r, s)][key] for s in ss])
            d1 = 100 * (m("canonX_ref", stress) - m("orbital", stress))
            d2 = 100 * (m("canon_ref", stress) - m("orbital", stress))
            d3 = 100 * (m("canon_max", sts) - m("canon_ref", sts))
            d4 = 100 * (m("phasemap_ml", sts) -
                        np.mean([max(grid[(p, r, s)]["euclid"], grid[(p, r, s)]["orbital"])
                                 for s in sts]))
            rows.append((p, r, d1, d2, d3, d4))
            L.append(f"| {p} | {r:g} | {d1:+.1f} | {d2:+.1f} | {d3:+.1f} | {d4:+.1f} |")

    # ---- 判定 ----
    L.append("\n## 预注册判定\n")
    L.append("| 预测 | 内容 | 判定 |")
    L.append("|---|---|---|")
    best = max(rows, key=lambda t: t[2])[2]
    L.append(f"| P-a | 存在区域使 canonX_ref 与 orbital 差 ≤1pt | "
             f"{'通过' if best >= -1.0 else '不通过'}（最优 {best:+.1f} PP，r=0.25） |")
    mono = all(rows[i + 1][2] <= rows[i][2] + 0.6
               for i in range(len(rows) - 1) if rows[i][0] == rows[i + 1][0])
    L.append(f"| P-b | 规范化代价随 r 单调扩大 | {'通过' if mono else '部分（见上表）'} |")
    sel = min(r3 for _, _, _, _, r3, _ in rows)
    L.append(f"| P-c | canon_max 劣于 canon_ref ≥10pt（选择偏倚） | "
             f"{'通过' if sel <= -10 else '不通过'}（最差 {sel:+.1f} PP） |")
    d4min = min(r4 for *_, r4 in rows)
    L.append(f"| P-d | phasemap_ml ≥ max(端点)−1pt | {'通过' if d4min >= -1.0 else '不通过'}"
             f"（最差 {d4min:+.1f} PP） |")
    c16 = [r2 for p, r, _, r2, _, _ in rows if p == 16 and r == 2]
    c64 = [r2 for p, r, _, r2, _, _ in rows if p == 64 and r == 2]
    if c16 and c64:
        L.append(f"| P-e | p=64 规范化代价 ≥ p=16 | {'通过' if c64[0] <= c16[0] else '不通过'}"
                 f"（r=2：p16 {c16[0]:+.1f} vs p64 {c64[0]:+.1f}） |")

    open(os.path.join(BASE, "04_results", "tables", "T8_b18_synth.md"), "w",
         encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for pi, p in enumerate(ps):
        ax[pi].set_title(f"p={p}: accuracy vs σ_θ")
        for key, ls, lw in [("orbital", "-", 2.2), ("phasemap_ml", "-", 2.2),
                            ("canonX_ref", "--", 1.8), ("canon_ref", "--", 1.8),
                            ("canon_max", ":", 1.5), ("euclid", "-.", 1.5)]:
            for r in rs:
                ys = [100 * grid[(p, r, s)][key] for s in sts]
                ax[pi].plot(sts, ys, ls, lw=lw, color=None,
                            label=f"{key}" if r == rs[0] else None,
                            alpha=0.45 + 0.18 * rs.index(r))
        ax[pi].set_xlabel(r"$\sigma_\theta$"); ax[pi].set_ylabel("acc %")
        ax[pi].legend(fontsize=6); ax[pi].grid(alpha=.3)
    ax[2].set_title("canonicalization cost vs noise ratio r")
    for pi, p in enumerate(ps):
        xs = rs
        y1 = [100 * (np.mean([grid[(p, r, s)]["canonX_ref"] for s in stress]) -
                     np.mean([grid[(p, r, s)]["orbital"] for s in stress])) for r in rs]
        y2 = [100 * (np.mean([grid[(p, r, s)]["canon_ref"] for s in stress]) -
                     np.mean([grid[(p, r, s)]["orbital"] for s in stress])) for r in rs]
        ax[2].plot(range(len(xs)), y1, "o-", color=f"C{pi}", label=f"p={p} canonX_ref−orbital")
        ax[2].plot(range(len(xs)), y2, "s--", color=f"C{pi}", label=f"p={p} canon_ref−orbital")
    ax[2].set_xticks(range(len(rs))); ax[2].set_xticklabels([f"{r:g}" for r in rs])
    ax[2].set_xlabel("noise/signal energy ratio r"); ax[2].set_ylabel("ΔPP vs orbital")
    ax[2].axhline(0, color="k", lw=.8); ax[2].legend(fontsize=6); ax[2].grid(alpha=.3)
    plt.tight_layout()
    fp = os.path.join(BASE, "04_results", "figures", "F8_b18_synth.png")
    plt.savefig(fp, dpi=150)
    print("\n[figure]", fp)


def real_report(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]
    keys = sorted({k for r in recs for k in r["acc"]})
    snrs = sorted({r["snr"] for r in recs})
    sigs = sorted({round(r["sigma"], 2) for r in recs})
    L = ["# T9：B18 真实 IQ —— 规范化的割迹代价（自然精度差 Δ）\n",
         "> 臂：raw_nat（自然训练）/ canon_nat（规范化后训练，严格 U(1) 不变）/ raw_aug（旋转增广）。"
         "编码器统一 AmcCNNWrap，预算一致。\n",
         "| SNR | σ_θ | " + " | ".join(k for k in keys if not k.startswith("ident")) + " |",
         "|" + "---|" * (len([k for k in keys if not k.startswith("ident")]) + 2)]
    for sl in snrs:
        for sg in sigs:
            sel = [r for r in recs if r["snr"] == sl and round(r["sigma"], 2) == sg]
            if not sel:
                continue
            vals = []
            for k in keys:
                if k.startswith("ident"):
                    continue
                v = np.concatenate([np.asarray(r["acc"][k], float) for r in sel])
                vals.append(f"{100*v.mean():.1f}")
            L.append(f"| {sl} | {sg:.2f} | " + " | ".join(vals) + " |")
    L.append("\n## 主终点：规范化代价 Δ（canon_nat − raw_nat，自然格 σ_θ=0）\n")
    L.append("| SNR | Δ_euclid | Δ_orbital | Δ_phML |")
    L.append("|---|---|---|---|")
    for sl in snrs:
        sel = [r for r in recs if r["snr"] == sl and round(r["sigma"], 2) == 0.0]
        if not sel:
            continue
        out = []
        for h in ["euclid", "orbital", "phML"]:
            a = np.concatenate([np.asarray(r["acc"][f"canon_nat_{h}"], float) for r in sel])
            b = np.concatenate([np.asarray(r["acc"][f"raw_nat_{h}"], float) for r in sel])
            out.append(f"{100*(a.mean()-b.mean()):+.2f}")
        L.append(f"| {sl} | " + " | ".join(out) + " |")
    al = [r for r in recs if "align" in r and "ident" in r["align"]]
    if al:
        L.append("\n## 规范化稳定性（离割迹距离 |z^H v|/(‖z‖‖v‖)，越大越安全）\n")
        L.append("| SNR | 值 |")
        L.append("|---|---|")
        for sl in snrs:
            v = [r["align"]["ident"] for r in al if r["snr"] == sl]
            if v:
                L.append(f"| {sl} | {np.mean(v):.3f} |")
    open(os.path.join(BASE, "04_results", "tables", "T9_b18_real.md"), "w",
         encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=None)
    ap.add_argument("--real", default=None)
    a = ap.parse_args()
    if a.synth and os.path.exists(a.synth):
        synth_report(a.synth)
    if a.real and os.path.exists(a.real):
        real_report(a.real)
