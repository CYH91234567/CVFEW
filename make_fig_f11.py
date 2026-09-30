"""F11：B26 收缩对齐 MSE 相图（λ × σ_θ，B28 修正口径）。

四面板：
  (a) soft-full-oracle 方向 MSE 相图（"边际化"端点；λ≥λ*≈0.44 跨 σ_θ 近不变）
  (b) 最优自参考估计器 MSE 相图（对齐族的 best case）
  (c) σ_θ 全距 vs λ（crossover λ*≈0.442 可视化）
  (d) 截断族 U 形（C2 窗口代表格 λ=3.90, σ_θ=π/6）：MSE vs t
输出：04_results/figures/F11_b26_mse_phase.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
res = json.load(open(os.path.join(BASE, "04_results", "logs", "B26_shrinkage.json")))
rows = res["v3"]
claims = res["verdict"]["v3_claims"]

lams = sorted({r["lam"] for r in rows})
sts = sorted({r["st"] for r in rows})
SELFREF = ["self1", "self2", "self3", "self5", "self10", "l1pca30", "soft_vm_self"]


def grid(fn):
    G = np.full((len(sts), len(lams)), np.nan)
    for r in rows:
        G[sts.index(r["st"]), lams.index(r["lam"])] = fn(r)
    return G


G_soft = grid(lambda r: r["soft_full_oracle"])
G_best = grid(lambda r: min(r[k] for k in SELFREF))
G_mean = grid(lambda r: r["mean"])

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
vmin, vmax = 0.35, 2.0

ax = axes[0, 0]
im = ax.imshow(G_soft, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
               cmap="viridis")
ax.set_title("(a) soft-full (oracle ref, posterior mean)\n'marginalize' endpoint",
             fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(G_best, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
               cmap="viridis")
ax.set_title("(b) best self-referencing estimator\n(truncated/hard 'select' family)",
             fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)

for ax in (axes[0, 0], axes[0, 1]):
    ax.set_yticks(range(len(sts)))
    ax.set_yticklabels([f"{s:.2f}" for s in sts])
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:.2f}" for l in lams], rotation=45, fontsize=7)
    ax.set_ylabel(r"$\sigma_\theta$ (rad)")
    ax.set_xlabel(r"$\lambda = a^2/\sigma^2$ (per-sample ref SNR)")

ax = axes[1, 0]
lam_axis = [c["lam"] for c in claims["detail"]]
sf = [c["C6_softfull_range"] for c in claims["detail"]]
bs = [c["C6_bestself_range"] for c in claims["detail"]]
ax.semilogx(lam_axis, sf, "o-", label="soft-full $\\sigma_\\theta$-range", color="tab:green")
ax.semilogx(lam_axis, bs, "s-", label="best self-ref range", color="tab:red")
ax.axvline(claims["C6_crossover_lam"], ls="--", color="k", lw=1,
           label=f"crossover $\\lambda^*\\approx${claims['C6_crossover_lam']:.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"MSE range across $\sigma_\theta\in\{\pi/6..\pi\}$")
ax.set_title("(c) $\\sigma_\\theta$-invariance of soft-full holds for $\\lambda\\geq\\lambda^*$\n"
             "(B28 correction: reverses below $\\lambda^*$)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
# C2 窗口代表格：λ=3.90, σ_θ=π/6 的截断族 U 形
cell = next(r for r in rows if abs(r["lam"] - 3.899) < 1e-3 and abs(r["st"] - np.pi / 6) < 1e-3)
ts = [0, 1, 2, 3, 5, 10, 30]
mses = [cell["mean"], cell["self1"], cell["self2"], cell["self3"],
        cell["self5"], cell["self10"], cell["l1pca30"]]
ax.plot(ts, mses, "o-", color="tab:purple")
ax.axhline(cell["soft_full_oracle"], color="tab:green", ls="--",
           label=f"soft-full = {cell['soft_full_oracle']:.3f}")
t_star = min(zip(ts, mses), key=lambda p: p[1])
ax.axvline(t_star[0], color="tab:purple", ls=":", alpha=0.7,
           label=f"optimal truncation t* = {t_star[0]}")
ax.set_xlabel("alignment truncation t (coord-ascend iters; 0=naive mean)")
ax.set_ylabel("direction MSE")
ax.set_title("(d) truncation U-shape (C2 window, $\\lambda$=3.90, $\\sigma_\\theta$=$\\pi$/6):\n"
             "'approximate alignment = regularization'", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("F11  B26 shrinkage-alignment MSE phase diagram (B28-corrected; "
             "B26_shrinkage.json)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(BASE, "04_results", "figures", "F11_b26_mse_phase.png")
fig.savefig(out, dpi=160)
print("saved", out)
