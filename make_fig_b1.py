"""B1 诊断图：三值门证据（相位结构 + 探针下降 + 载波参考）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
MOD = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

r = json.load(open(os.path.join(RES, "logs", "B1_diagnostic.json")))
p = json.load(open(os.path.join(RES, "logs", "B1_phase_structure.json")))

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle(f"B1 Diagnostic on RML2016.10a (SNR>=0 subset) — Gate: {r['gate']['gate']}", fontsize=13)

# (a) 探针：特征模态 nat vs rand-matched
ax = axes[0, 0]
modes = ["rip", "amp", "apc"]
nat = [100 * r["probe"]["per_feature"][m]["acc"] for m in modes]
rnd = [100 * r["probe"]["per_feature"][m]["acc_randphase_matched"] for m in modes]
x = np.arange(3)
ax.bar(x - 0.18, nat, 0.36, label="natural frames")
ax.bar(x + 0.18, rnd, 0.36, label="global-phase randomized (matched retrain)")
ax.set_xticks(x); ax.set_xticklabels(["Re,Im", "amplitude", "amp+cos/sin$\\Phi$"])
ax.set_ylabel("linear probe acc (%)"); ax.legend(); ax.set_title("(a) Global-phase info (matched protocol)")

# (b) 逐类 matched drop
ax = axes[0, 1]
pc = r["probe"]["per_class"]
order = sorted(pc.keys(), key=lambda c: pc[c]["drop_matched"])
vals = [100 * pc[c]["drop_matched"] for c in order]
cols = ["#d62728" if v > 1 else ("#2ca02c" if v < -1 else "#7f7f7f") for v in vals]
ax.barh(range(len(order)), vals, color=cols)
ax.set_yticks(range(len(order))); ax.set_yticklabels([MOD[int(c)] for c in order])
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("matched drop (pts)")
ax.set_title("(b) Per-class: red=phase informative, green=phase nuisance")

# (c) R̄ 同/跨类（加权合向量长度），按类平均跨SNR档
ax = axes[1, 0]
rb = p["rbar"]
cs = sorted(rb.keys(), key=lambda c: np.mean([b["R_same_w"] for b in rb[c].values() if b]))
same = [np.mean([b["R_same_w"] for b in rb[c].values() if b]) for c in cs]
cross = [np.mean([b["R_cross_w"] for b in rb[c].values() if b]) for c in cs]
x = np.arange(len(cs))
ax.bar(x - 0.18, same, 0.36, label="same-class pairs")
ax.bar(x + 0.18, cross, 0.36, label="cross-class pairs")
ax.set_xticks(x); ax.set_xticklabels([MOD[int(c)] for c in cs], rotation=60, fontsize=7)
ax.set_ylabel("pairwise inner-product phase $\\bar R$ (weighted)")
ax.legend(); ax.set_title("(c) Same- vs cross-class phase coherence")

# (d) DC载波：相对幅度 vs 相位集中度
ax = axes[1, 1]
car = r["carrier"]["per_class"]
for c in car:
    ax.scatter(car[c]["dc_rel_mean"], car[c]["dc_phase_R"], s=28)
    ax.annotate(MOD[int(c)], (car[c]["dc_rel_mean"], car[c]["dc_phase_R"]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("DC component relative magnitude |E[z]|/RMS")
ax.set_ylabel("DC phase concentration R across frames")
ax.set_title("(d) Carrier reference: analog classes carry stable phase reference")

plt.tight_layout()
out = os.path.join(RES, "figures", "B1_diagnostic.png")
plt.savefig(out, dpi=160)
print("saved", out)
