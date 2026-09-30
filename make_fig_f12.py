"""F12：B29c 双盆分布 + B29b 选点信号校准 + 筛选信号对比（第九轮）。

四面板：
  (a) 12 runs valsel 直方图 + 双峰（好盆中心 96.3 / 坏盆 77.6 / max 98.5）
  (b) B29b 选点信号校准：三变体 Spearman(val,test) + 饱和度（max）
  (c) valsel vs endpoint 配对散点（对角线；选点收益 + 双群着色）
  (d) 早期筛选信号证伪：vt@800 vs final（AUC 0.16=无/反向）vs
      cancel_ratio（AUC 1.00，endpoint 测）
输出：04_results/figures/F12_b29_basin.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
rc = json.load(open(os.path.join(BASE, "04_results", "logs", "B29c_sca_v3.json")))
rb = json.load(open(os.path.join(BASE, "04_results", "logs", "B29b_valsel_calibration.json")))
rv = json.load(open(os.path.join(BASE, "04_results", "logs", "B29c_verdict.json")))
SIG = (0.0, np.pi / 3, np.pi)


def hm(v, w):
    if w not in v["eval"]:
        return None
    return 100 * float(np.mean([v["eval"][w][f"sigma_{s:.2f}"]["orbital"] for s in SIG]))


runs = {k: v for k, v in rc["runs"].items() if k.startswith("cx_v2_vt")}
keys = sorted(runs)
vs = np.array([hm(v, "valsel") for v in runs.values()])
ep = np.array([hm(v, "endpoint") for v in runs.values()])
good = vs >= 93
cancel = np.array([v["info"]["cancel_ratio"] for v in runs.values()])
vc800 = np.array([dict(v["info"]["val_curve"])[800] for v in runs.values()])

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

# (a) 双盆直方图
ax = axes[0, 0]
bins = np.linspace(65, 100, 18)
ax.hist(vs[good], bins=bins, color="tab:green", alpha=0.75, label="good basin (n=4)")
ax.hist(vs[~good], bins=bins, color="tab:red", alpha=0.75, label="bad basin (n=8)")
ax.axvline(97.0, color="k", ls="--", lw=1, label="SC-A threshold 97.0")
ax.axvline(vs[good].mean(), color="tab:green", ls=":", lw=1.5,
           label=f"good center {vs[good].mean():.1f}")
ax.axvline(vs[~good].mean(), color="tab:red", ls=":", lw=1.5,
           label=f"bad center {vs[~good].mean():.1f}")
ax.set_xlabel("valsel accuracy (%, orbital head, 3$\\sigma$ mean)")
ax.set_ylabel("# runs")
ax.set_title(f"(a) Bimodal basin distribution (B29c, n=12, val_train selection)\n"
             f"best {vs.max():.1f} · good-basin rate {good.mean():.0%} · "
             f"$\\sigma$-flatness {100*rv['verdict']['flat_valsel_max']:.1f}pt",
             fontsize=10)
ax.legend(fontsize=7)

# (b) 选点信号校准（B29b）
ax = axes[0, 1]
names = ["val_5shot\n(2-way holdout,\nB29 used)", "val_1shot\n(2-way 1-shot)",
         "val_train\n(train-class\n5-way)"]
sp = [rb["trajs"][k]["align"]["val_5shot"]["spearman"] for k in ("i11_f0", "i11_f12", "i23_f0")]
sp1 = [rb["trajs"][k]["align"]["val_1shot"]["spearman"] for k in ("i11_f0", "i11_f12", "i23_f0")]
spt = [rb["trajs"][k]["align"]["val_train"]["spearman"] for k in ("i11_f0", "i11_f12", "i23_f0")]
mx = [rb["trajs"][k]["align"]["val_5shot"]["max"] for k in ("i11_f0", "i11_f12", "i23_f0")]
mx1 = [rb["trajs"][k]["align"]["val_1shot"]["max"] for k in ("i11_f0", "i11_f12", "i23_f0")]
mxt = [rb["trajs"][k]["align"]["val_train"]["max"] for k in ("i11_f0", "i11_f12", "i23_f0")]
x = np.arange(3)
ax.bar(x - 0.2, [np.mean(s) for s in (sp, sp1, spt)], 0.4,
       yerr=[np.std(s) for s in (sp, sp1, spt)], capsize=3,
       color=["tab:orange", "tab:purple", "tab:blue"], label="Spearman(val, test)")
ax.axhline(0, color="k", lw=0.8)
ax2 = ax.twinx()
ax2.plot(x + 0.2, [np.mean(m) for m in (mx, mx1, mxt)], "s--", color="gray",
         label="max val (saturation)")
ax2.axhline(0.99, color="gray", ls=":", lw=0.8)
ax2.set_ylim(0.5, 1.05)
ax2.set_ylabel("max val accuracy (saturation)", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Spearman(val, test) across trajectory")
ax.set_title("(b) Selection-signal calibration (B29b, 3 trajectories)\n"
             "val_train: aligned & unsaturated; 2-way holdout: saturates to 1.0",
             fontsize=10)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")

# (c) valsel vs endpoint 配对散点
ax = axes[1, 0]
ax.scatter(ep[good], vs[good], c="tab:green", s=60, label="good basin")
ax.scatter(ep[~good], vs[~good], c="tab:red", s=60, label="bad basin")
lim = [60, 102]
ax.plot(lim, lim, "k--", lw=1)
for e, v, k in zip(ep, vs, keys):
    ax.annotate(k.split("_i")[1].split("_")[0] + "/" + k.split("_f")[1],
                (e, v), fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("endpoint accuracy (%)")
ax.set_ylabel("valsel accuracy (val_train selection)")
ax.set_title(f"(c) Selection gain valsel−endpoint: mean "
             f"{np.mean(vs-ep):+.1f}pt\n(bad basins recover +6..+12pt)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (d) 筛选信号对比：vt@800（证伪）vs cancel_ratio
ax = axes[1, 1]
ax.scatter(vc800[good], vs[good], c="tab:green", s=60, marker="o", label="cancel_rt good")
ax.scatter(vc800[~good], vs[~good], c="tab:red", s=60, marker="o", label="cancel_rt bad")
ax.set_xlabel("vt@800 (train-class val at step 800)")
ax.set_ylabel("final valsel (%)")
ax.set_title("(d) Early screening signal refuted: vt@800 has no predictive power\n"
             "(AUC 0.16; good/bad overlap完全) — basin is init×stream, not "
             "early-trainable", fontsize=10)
ax.grid(alpha=0.3)
ax.annotate("cancel_ratio (post-train)\nseparates perfectly: AUC=1.00",
            xy=(0.72, 97), fontsize=8, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue"),
            xytext=(0.745, 82))
ax.legend(fontsize=8)

fig.suptitle("F12  B29c basin distribution + selection-signal calibration + "
             "screening refutation", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(BASE, "04_results", "figures", "F12_b29_basin.png")
fig.savefig(out, dpi=160)
print("saved", out)
