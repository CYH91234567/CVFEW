"""F13：B29d 盆筛选验证汇总（cancel/vt AUC 时程 + 28 runs 散点）。

(a) cancel_ratio 与 vt 的逐时点 AUC（好盆识别，B29d 16 runs）
(b) cancel@800 vs final valsel（B29c+B29d 28 runs）——信号可检测但重叠大
(c) valsel 分布（B29c+B29d 合并 28 runs，双峰/三层结构）
(d) 选点信号校准与筛选结果的汇总结论面板
输出：04_results/figures/F13_b29d_screening.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
rd = json.load(open(os.path.join(BASE, "04_results", "logs", "B29d_screening.json")))
rc = json.load(open(os.path.join(BASE, "04_results", "logs", "B29c_sca_v3.json")))
SIG = (0.0, np.pi / 3, np.pi)


def hm(v):
    return 100 * float(np.mean([v["eval"]["valsel"][f"sigma_{s:.2f}"]["orbital"]
                                for s in SIG]))


def cancel_at(v, it):
    return dict((d["it"], d["cancel"]) for d in v["info"]["diag"]).get(it)


d_runs = list(rd["runs"].values())
c_runs = [v for k, v in rc["runs"].items() if k.startswith("cx_v2_vt")]

vs_d = np.array([hm(v) for v in d_runs])
good_d = vs_d >= 93


def auc(scores, good):
    s = np.asarray(scores, float)
    if good.sum() == 0 or good.sum() == len(good):
        return np.nan
    return mannwhitneyu(s[good], s[~good]).statistic / (good.sum() * (~good).sum())


its = [100, 200, 400, 600, 800, 1000, 1600, 2400, 3200]
cr_auc = [auc([cancel_at(v, it) for v in d_runs], good_d) for it in its]
vt_vals = [[dict((d["it"], d["vt"]) for d in v["info"]["diag"]).get(it)
            for v in d_runs] for it in its]
vt_auc = [auc(vv, good_d) for vv in vt_vals]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

ax = axes[0, 0]
ax.plot(its, cr_auc, "o-", color="tab:blue", label="cancel_ratio AUC")
ax.plot(its, vt_auc, "s-", color="tab:red", label="vt (train-class val) AUC")
ax.axhline(0.5, color="k", ls=":", lw=1, label="chance (0.5)")
ax.axhline(0.8, color="tab:blue", ls="--", lw=1, alpha=0.5, label="SC-S1 threshold (0.8)")
ax.set_xscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel("AUC (good-basin detection, n_good=2)")
ax.set_title("(a) Screening-signal AUC over training (B29d, 16 runs, 2 good)\n"
             "cancel_ratio: detectable but weak; vt: at/below chance", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.02)

# (b) cancel@800 vs final valsel（28 runs 合并）
ax = axes[0, 1]
cr_d = np.array([cancel_at(v, 800) for v in d_runs])
cr_c = np.array([v["info"]["cancel_ratio"] for v in c_runs])
vs_c = np.array([hm(v) for v in c_runs])
good_c = vs_c >= 93
ax.scatter(cr_d, vs_d, c=["tab:green" if g else "tab:red" for g in good_d],
           s=55, marker="o", label="B29d (cancel@800)")
ax.scatter(cr_c, vs_c, c=["tab:green" if g else "tab:red" for g in good_c],
           s=55, marker="s", alpha=0.75, label="B29c (cancel@endpoint)")
for x0, y0, k in zip(cr_d, vs_d, rd["runs"]):
    ax.annotate(k.split("_i")[1], (x0, y0), fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("cancel_ratio (B29d: @step 800; B29c: @endpoint)")
ax.set_ylabel("final valsel (%)")
ax.set_title("(b) Signal detectable but not deployable: 28 runs pooled\n"
             "good basins cluster high, but bad basins overlap in 0.49-0.54",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) 28 runs valsel 分布
ax = axes[1, 0]
allv = np.concatenate([vs_d, vs_c])
ax.hist([allv], bins=np.linspace(60, 102, 22), color="tab:gray", alpha=0.6,
        label=f"all 28 runs (mean {allv.mean():.1f})")
ax.hist(allv[allv >= 93], bins=np.linspace(60, 102, 22), color="tab:green",
        alpha=0.8, label=f"good ≥93 (n={int((allv>=93).sum())}, "
                         f"center {allv[allv>=93].mean():.1f})")
ax.axvline(97.0, color="k", ls="--", lw=1, label="SC-A threshold 97.0")
ax.axvline(allv.max(), color="tab:green", ls=":", lw=1.5,
           label=f"best {allv.max():.1f}")
ax.set_xlabel("valsel accuracy (%)")
ax.set_ylabel("# runs")
ax.set_title("(c) B29c+B29d pooled: bimodal/tri-modal basin\n"
             f"good rate {(allv>=93).mean():.0%} · σ-flatness ≤0.95pt in every run",
             fontsize=10)
ax.legend(fontsize=8)

# (d) 结论面板
ax = axes[1, 1]
ax.axis("off")
txt = (
    "Basin screening: REFUTED as a protocol\n"
    "────────────────────────────────\n"
    "vt@800: AUC 0.07-0.16 (at or below chance) — refuted\n"
    "cancel_ratio@800: AUC 0.86 (SC-S1 pass) but\n"
    "  median split → 12.5% good rate (SC-S2 fail, = baseline)\n"
    "  ⇒ signal is detectable, not deployable\n\n"
    "Basin = init × stream (irreducible under any\n"
    "train-time observable tested: val level, trajectory-\n"
    "internal val, loss, cancel_ratio, fisher)\n\n"
    "Bimodal structure is robust across 28 runs:\n"
    f"  good center 96-98 (max 98.5) · bad center 77-81\n"
    "  equivariance certificate (σ-flatness) holds in\n"
    "  every basin ⇒ failure is optimization/init, not\n"
    "  the equivariant structure itself")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9.2,
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))
ax.set_title("(d) Honest conclusion", fontsize=10)

fig.suptitle("F13  B29d screening-protocol validation + 28-run pooled basin structure",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(BASE, "04_results", "figures", "F13_b29d_screening.png")
fig.savefig(out, dpi=160)
print("saved", out)
