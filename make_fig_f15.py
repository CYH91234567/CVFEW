"""F15：B31 不变结构读出结果图。
(a) 四臂分布（autocorr/nopool/power/gated 逐 init 散点）；
(b) autocorr vs gated 配对差；
(c) σ 平坦性对比（构造性不变证书 0.01pt）。
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei UI", "NSimSun", "FangSong"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, np.pi / 3, np.pi)
ARM_HEAD = {"autocorr": "euclid", "power": "euclid", "nopool": "orbital", "gated": "orbital"}
LABELS = {"autocorr": "autocorr（不变结构）", "power": "power（幅度不变）",
          "nopool": "nopool（序列拉平）", "gated": "gated（对照）"}


def load(name):
    p = os.path.join(LOG, f"B30_{name}.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for k, v in json.load(open(p))["runs"].items():
        e = v["eval"].get("valsel")
        if e:
            out[k.split("_i")[-1]] = 100 * np.mean(
                [e[f"sigma_{s:.2f}"][ARM_HEAD[name]] for s in SIG])
    return out


def main():
    arms = ["gated", "power", "autocorr", "nopool"]
    data = {a: load(a) for a in arms}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ys = {a: [data[a][i] for i in sorted(data[a])] for a in arms}
    for ai, a in enumerate(arms):
        ax.scatter([ai] * len(ys[a]), ys[a], s=30, alpha=.85, zorder=3,
                   label=f"{a}: μ={np.mean(ys[a]):.1f} σ={np.std(ys[a], ddof=1):.1f}")
    ax.axhline(93, color="k", ls="--", lw=1)
    ax.text(3.4, 93.6, "好盆 93", fontsize=8, ha="right")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([LABELS[a] for a in arms],
                                                        rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("valsel 三头均值 (%)")
    ax.set_title("(a) 逐 init 分布：autocorr 均值最高且方差腰斩\n"
                 "（μ 82.9→85.8，σ 10.3→5.4；min 67.3→71.8）", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    g, ac = ys["gated"], ys["autocorr"]
    d = [a - b for a, b in zip(ac, g)]
    ax.hist(d, bins=12, color="C2", alpha=.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.mean(d), color="C3", ls="--", lw=1.5, label=f"配对均值 {np.mean(d):+.2f}pt")
    ax.set_xlabel("autocorr − gated（同 init×流，pt）"); ax.set_ylabel("init 数")
    ax.set_title(f"(b) 配对差（胜率 {np.mean([x > 0 for x in d]):.0%}，p=0.35；\n"
                 f"主效应=方差与最差情形改善）", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[2]
    names, flats = [], []
    for a in arms:
        sp = []
        for v in json.load(open(os.path.join(LOG, f"B30_{a}.json")))["runs"].values():
            e = v["eval"].get("valsel")
            if e:
                vals = [100 * e[f"sigma_{s:.2f}"][ARM_HEAD[a]] for s in SIG]
                sp.append(max(vals) - min(vals))
        if sp:
            names.append(LABELS[a].split("（")[0]); flats.append((np.mean(sp), np.max(sp)))
    x = np.arange(len(names))
    ax.bar(x - .18, [f[0] for f in flats], width=.36, label="mean", color="C0")
    ax.bar(x + .18, [f[1] for f in flats], width=.36, label="max", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("σ 轴平坦残差 (pt)"); ax.set_yscale("log")
    ax.axhline(0.1, color="k", ls="--", lw=1)
    ax.text(len(names) - .5, 0.12, "0.1pt（float 级）", fontsize=7, ha="right")
    ax.set_title("(c) σ 配对平坦性：不变读出 0.01pt（构造性证书）", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle("B31 不变结构读出：相位差/矩统计量=保持信息的不变化（64 runs，含 B30 gated 配对）",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "04_results", "figures", "F15_b31_invariant.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
