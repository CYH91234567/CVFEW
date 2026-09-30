"""F14：B30 五臂盆结构图（负结果可视化）。
(a) 五臂逐 init 散点（配对连线 orbit-gated，同 init×流）；
(b) 臂内分布小提琴/箱线（双盆消灭但天花板压低）；
(c) 机制示意水位带：gated 坏盆 67-78 ≈ orbit 全域（对齐=坏盆水位）。
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


def load(name):
    d = json.load(open(os.path.join(LOG, f"B30_{name}.json")))
    out = {}
    for k, v in d["runs"].items():
        e = v["eval"].get("valsel")
        if e:
            out[k.split("_i")[-1]] = 100 * sum(
                e[f"sigma_{s:.2f}"]["orbital"] for s in SIG) / 3
    return out


def main():
    arms = ["gated", "orbit", "hybrid", "orbitL0.3", "learnlam"]
    labels = ["gated (对照)", "orbit λ=1", "hybrid", "orbit λ=0.3", "learnlam"]
    data = {a: load(a) for a in arms}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) 逐 init 散点 + orbit-gated 配对连线
    ax = axes[0]
    ys = {a: [data[a][i] for i in sorted(data[a])] for a in arms}
    for ai, a in enumerate(arms):
        ax.scatter([ai] * len(ys[a]), ys[a], s=26, alpha=.85, zorder=3)
    g, o = ys["gated"], ys["orbit"]
    for i in range(min(len(g), len(o))):
        ax.plot([0, 1], [g[i], o[i]], color="gray", alpha=.45, lw=.8, zorder=1)
    ax.axhspan(67, 78, color="C3", alpha=.08, zorder=0)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("valsel 三头均值 (%)"); ax.set_title("(a) 逐 init（灰线=同 init 配对）\n红带=gated 坏盆水位 67–78")

    # (b) 分布
    ax = axes[1]
    bp = ax.boxplot([ys[a] for a in arms], patch_artist=True, showfliers=False)
    for bi, b in enumerate(bp["boxes"]):
        b.set_facecolor(f"C{bi}"); b.set_alpha(.5)
    ax.axhline(93, color="k", ls="--", lw=1); ax.text(4.6, 93.5, "好盆线 93", fontsize=8, ha="right")
    ax.set_xticks(range(1, len(arms) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("(b) 臂内分布：sd 10.3→1.0–2.7（双盆消灭）\n但均值 82.9→69.0（天花板压低）")
    ax.set_ylabel("valsel (%)")

    # (c) 配对差直方图
    ax = axes[2]
    d = [o - g for o, g in zip(ys["orbit"], ys["gated"])]
    ax.hist(d, bins=10, color="C1", alpha=.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.mean(d), color="C3", ls="--", lw=1.5,
               label=f"均值 {np.mean(d):+.1f}pt")
    ax.set_xlabel("orbit − gated（配对，pt）"); ax.set_ylabel("init 数")
    ax.set_title(f"(c) 配对差（Wilcoxon p=4.3e−4）\n胜率 {np.mean([x > 0 for x in d]):.0%}")
    ax.legend(fontsize=8)
    fig.suptitle("B30 轨道池化：分析式相位同步消灭双盆但破坏帧内时间相位信息（80 runs）",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "04_results", "figures", "F14_b30_basins.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
