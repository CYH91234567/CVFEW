"""B17 分析：压力格主终点 + 配对检验 + κ̂ 可辨识性 + 图表。

用法: python analyze_b17.py --src 04_results/logs/B17_pilot_p1.json [--stress-min 1.047]
输出: 04_results/tables/T7_b17_head.md, 04_results/figures/F7_b17_heads.png
"""
import argparse, json, os, sys
import numpy as np

try:
    from scipy.stats import wilcoxon, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def boot_ci(d, n_boot=4000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(d_a, d_b):
    d = np.asarray(d_a) - np.asarray(d_b)
    lo, hi = boot_ci(d)
    p = None
    if HAVE_SCIPY and len(d) > 5:
        try:
            p = float(wilcoxon(d).pvalue)
        except Exception:
            p = None
    return float(100 * d.mean()), float(100 * lo), float(100 * hi), p


def collect(records, keys):
    """按 sigma 汇总：每方法的逐 episode 精度（跨 split/seed 拼接）。"""
    out = {}
    for r in records:
        sg = round(r["sigma"], 3)
        for k in keys:
            if k in r["acc"]:
                out.setdefault((sg, k), []).extend(np.asarray(r["acc"][k], dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--stress-min", type=float, default=np.pi / 3)
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    d = json.load(open(a.src, encoding="utf-8"))
    recs = d["records"]
    meta = d.get("meta", {})
    keys = sorted({k for r in recs for k in r["acc"]})
    raw = collect(recs, keys)
    sigmas = sorted({round(r["sigma"], 3) for r in recs})
    stress = [s for s in sigmas if s >= a.stress_min - 1e-6]
    natural = [s for s in sigmas if s <= 1e-6]

    def arr(sg_list, k):
        v = []
        for s in sg_list:
            v.extend(raw.get((s, k), []))
        return np.asarray(v)

    # ---------- 主表 ----------
    lines = []
    lines.append(f"# T7：B17 商空间分类头 × 学习到的表示（源：{os.path.basename(a.src)}）\n")
    lines.append(f"> 压力格 σ_θ≥{a.stress_min:.2f}（共 {len(stress)} 格：{stress}）；"
                 f"自然格 σ_θ=0。配对 episode，PP 单位，bootstrap 95% CI。\n")
    lines.append("| 方法 | 自然格 | 压力格 | Δ(压力−自然) | 参数量 | 等变δ |")
    lines.append("|---|---|---|---|---|---|")
    rows = []
    for k in keys:
        if k.endswith("_phML_oracle"):
            continue
        nat, st = arr(natural, k), arr(stress, k)
        if len(nat) == 0 or len(st) == 0:
            continue
        nm = k.split("_")[0] if k.startswith(("amc", "real", "cx", "ident")) else k
        prm = meta.get(nm, {}).get("params", "")
        dl = meta.get(nm, {}).get("equiv_delta", "")
        rows.append((100 * st.mean(), k, 100 * nat.mean(), 100 * st.mean() - 100 * nat.mean(), prm, dl))
    for _, k, nat, st, dl_, prm in sorted(rows, reverse=True):
        nm = k.split("_")[0]
        pv = meta.get(nm, {}).get("params", "")
        ev = meta.get(nm, {}).get("equiv_delta", "")
        ev = f"{ev:.4f}" if isinstance(ev, float) else "—"
        lines.append(f"| `{k}` | {nat:.1f} | {st:.1f} | {dl_:+.1f} | {pv} | {ev} |")

    # ---------- 假设检验 ----------
    lines.append("\n## 假设检验（压力格，配对 episode）\n")
    lines.append("| # | 对照 | Δ (PP) | 95% CI | Wilcoxon p | 判定 |")
    lines.append("|---|---|---|---|---|---|")

    def cmp(name, ka, kb, thr=0.0, ge=True):
        if ka not in keys or kb not in keys:
            return None
        m, lo, hi, p = paired(arr(stress, ka), arr(stress, kb))
        ok = (m >= thr) if ge else (m <= thr)
        verdict = ("通过" if ok else "不通过") + ("" if lo > 0 or hi < 0 else "（CI 跨 0，不显著）")
        lines.append(f"| {name} | `{ka}` − `{kb}` | {m:+.2f} | [{lo:+.2f},{hi:+.2f}] | "
                     f"{'—' if p is None else f'{p:.2e}'} | {verdict} |")
        return m

    cmp("H1 结构vs学习不变性", "cx_orbital_orbital", "real_aug_euclid", 2.0)
    cmp("H1' 结构vs学习不变性(amc)", "cx_orbital_orbital", "amc_aug_euclid", 2.0)
    cmp("H2 同编码器换头", "cx_orbital_orbital", "cx_orbital_euclid", 8.0)
    cmp("H2' phML vs 欧氏头", "cx_orbital_phML", "cx_orbital_euclid", 8.0)
    cmp("H2'' 商头 vs 欧氏头(amc)", "amc_euclid_orbital", "amc_euclid_euclid", 0.0)
    # H3 自然格无损失
    for ka, kb in [("cx_orbital_orbital", "amc_euclid_euclid"), ("cx_orbital_orbital", "real_euclid_euclid")]:
        if ka in keys and kb in keys:
            m, lo, hi, p = paired(arr(natural, ka), arr(natural, kb))
            lines.append(f"| H3 自然格无损失 | `{ka}` − `{kb}` | {m:+.2f} | [{lo:+.2f},{hi:+.2f}] | "
                         f"{'—' if p is None else f'{p:.2e}'} | "
                         f"{'通过(≤3pt)' if abs(m) <= 3.0 else '不通过(>3pt)'} |")

    # ---------- κ̂ 可辨识性 ----------
    lines.append("\n## H4：轮廓 κ̂ 随注入 σ_θ 的辨识（嵌入层面）\n")
    kap_keys = sorted({k for r in recs if k in r for k in r if k.startswith("kappa_")})
    if kap_keys:
        lines.append("| 嵌入 | " + " | ".join(f"σ={s:.2f}" for s in sigmas) + " | Spearman |")
        lines.append("|---|---|")
        for kk in kap_keys:
            vals = []
            for s in sigmas:
                v = [r[kk] for r in recs if round(r["sigma"], 3) == s and kk in r]
                vals.append(np.mean(v) if v else np.nan)
            rho = np.nan
            if HAVE_SCIPY and len(sigmas) > 2:
                rho = float(spearmanr(sigmas, vals).statistic)
            lines.append(f"| `{kk}` | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {rho:.2f} |")

    lines.append("\n## 逐 σ_θ 曲线（PP）\n")
    hdr = "| 方法 | " + " | ".join(f"{s:.2f}" for s in sigmas) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(sigmas) + 1))
    for _, k, *_ in sorted(rows, reverse=True):
        cells = []
        for s in sigmas:
            v = arr([s], k)
            cells.append(f"{100*v.mean():.1f}" if len(v) else "—")
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")

    outdir = os.path.join(base, "04_results")
    tag = a.out_tag or os.path.basename(a.src).replace(".json", "")
    open(os.path.join(outdir, "tables", f"T7_b17_{tag}.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

    # ---------- 图 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for _, k, *_ in sorted(rows, reverse=True)[:10]:
        ys = [100 * arr([s], k).mean() for s in sigmas]
        ax[0].plot(sigmas, ys, marker="o", lw=1.6, label=k)
    ax[0].set_xlabel(r"injected $\sigma_\theta$ (rad)"); ax[0].set_ylabel("accuracy (%)")
    ax[0].set_title("B17: accuracy vs phase nuisance"); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
    # 压力格条形
    names = [k for _, k, *_ in sorted(rows, reverse=True)[:10]]
    means = [100 * arr(stress, k).mean() for k in names]
    errs = [100 * (arr(stress, k).std() / max(len(arr(stress, k)) ** .5, 1)) for k in names]
    ax[1].barh(range(len(names)), means, xerr=errs, color="#4C78A8")
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel(f"stress-grid accuracy (%, σ_θ≥{a.stress_min:.2f})")
    ax[1].set_title("B17 primary endpoint"); ax[1].grid(alpha=.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "figures", f"F7_b17_{tag}.png"), dpi=150)
    print("\n[figure]", os.path.join(outdir, "figures", f"F7_b17_{tag}.png"))


if __name__ == "__main__":
    main()
