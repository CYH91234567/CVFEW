"""B9：结果册——汇总图表与统计检验（全部本地numpy）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
LOG = os.path.join(RES, "logs")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")


def load(name):
    p = os.path.join(LOG, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f1_main_curves():
    rows = load("B5_grid.json")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (p, fade) in zip(axes, [(64, 0.0), (64, 0.5), (16, 0.0)]):
        for rho, ls in [(0.0, "-"), (0.3, "--"), (0.7, ":")]:
            rs = sorted([r for r in rows if r["tag"] == "main" and r["p"] == p
                         and r["rho"] == rho and r["k"] == 5 and r["fade_q"] == fade],
                        key=lambda r: r["sigma_th"])
            if not rs:
                continue
            x = [r["sigma_th"] for r in rs]
            for meth, col in [("euclid", "#d62728"), ("orbital", "#1f77b4"),
                              ("phasemap_ml", "#2ca02c")]:
                y = [100 * r["acc"][meth] for r in rs]
                lbl = f"{meth} rho={rho}" if meth != "phasemap_ml" else "PhaseMAP-ML"
                ax.plot(x, y, ls, color=col, lw=1.6, label=lbl)
        ax.set_xlabel("phase spread sigma_theta (rad)")
        ax.set_ylabel("5-way 5-shot acc (%)")
        ax.set_title(f"p={p}, fade q={fade}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("F1: PhaseFewSyn main grid (2000 paired episodes/condition)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F1_main_curves.png"), dpi=160, bbox_inches="tight")
    print("F1 saved")


def f2_adaptivity():
    rows = load("B5_grid.json")
    if not rows:
        return
    rs = sorted([r for r in rows if r["tag"] == "main" and r["p"] == 64 and r["rho"] == 0.3
                 and r["k"] == 5 and r["fade_q"] == 0.0], key=lambda r: r["sigma_th"])
    x = [r["sigma_th"] for r in rs]
    best = [100 * max(r["acc"]["euclid"], r["acc"]["orbital"]) for r in rs]
    worst = [100 * min(r["acc"]["euclid"], r["acc"]["orbital"]) for r in rs]
    ml = [100 * r["acc"]["phasemap_ml"] for r in rs]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(x, worst, best, alpha=0.18, color="gray",
                    label="endpoint envelope [min, max]")
    ax.plot(x, best, "k--", lw=1, label="best fixed endpoint")
    ax.plot(x, ml, "#2ca02c", lw=2.2, marker="o", ms=4, label="PhaseMAP-ML (adaptive)")
    ax.axhline(20, color="gray", lw=0.6)
    ax.set_xlabel("phase spread sigma_theta (rad)")
    ax.set_ylabel("acc (%)")
    ax.set_title("F2: adaptive estimator vs fixed endpoints\n(p=64, rho=0.3, 5-shot, 2000 episodes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F2_adaptivity.png"), dpi=160)
    print("F2 saved")


def f3_radioml():
    rows = load("B8_radioml.json")
    if not rows:
        return
    snrs = sorted(set(r["snr"] for r in rows))
    modes = [("none", 0.0), ("global", np.pi / 6), ("global", np.pi / 2)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, k in zip(axes, [5, 1]):
        for meth, col in [("euclid", "#d62728"), ("orbital", "#1f77b4"),
                          ("phasemap_ml", "#2ca02c"), ("tta_euclid", "#9467bd")]:
            ys = []
            for mode, strength in modes:
                for sl in snrs:
                    r = [q for q in rows if q["snr"] == sl and q["inject_mode"] == mode
                         and abs(q["inject_strength"] - strength) < 1e-6 and q["k"] == k]
                    ys.append(100 * r[0]["acc"][meth] if r else np.nan)
            ax.plot(range(len(ys)), ys, marker="o", ms=4, color=col, label=meth)
        ticks = [f"{'nat' if m == 'none' else 's=' + format(s, '.2f')}\nSNR{sl}"
                 for (m, s) in modes for sl in snrs]
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(f"{k}-shot")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("acc (%)")
    fig.suptitle("F3: RadioML RML2016.10a 5-way few-shot (600 episodes/cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F3_radioml.png"), dpi=160)
    print("F3 saved")


def f4_theory():
    d = load("B4_p4_risk.json")
    if not d:
        return
    x = [r["sigma_th"] for r in d]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, [100 * r["risk_euclid_numeric"] for r in d], "k-", lw=2, label="Euclid (analytic)")
    ax.plot(x, [100 * r["risk_euclid_mc"] for r in d], "ko", ms=4, label="Euclid (MC)")
    ax.plot(x, [100 * r["risk_orbital_mc"] for r in d], "#1f77b4", lw=2, label="Orbital (MC)")
    ax.axhline(50, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("phase spread sigma_theta (rad)")
    ax.set_ylabel("1-shot pairwise error (%)")
    ax.set_title("F4: P4 risk separation - Euclid rises to 50%, orbital invariant")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F4_theory.png"), dpi=160)
    print("F4 saved")


def f5_b6_encoder():
    d = load("B6_encoder.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    rs = [r for r in d["frozen_fewshot"] if r["setting"] == "complex_random_11class_5way"]
    x = np.arange(3)
    for meth, col, mk in [("euclid", "#d62728", "o"), ("orbital", "#1f77b4", "s"),
                          ("phasemap", "#2ca02c", "^")]:
        ax.plot(x, [100 * r["acc"][meth] for r in rs], marker=mk, color=col, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels(["natural", "inject pi/3", "inject pi"])
    ax.set_ylabel("acc (%)")
    ax.set_title("F5a: random-frozen equivariant trunk (11-class 5-way)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    rs = [r for r in d["frozen_fewshot"] if r["setting"].startswith("pretrained_split")
          and r["inject"] == "global_pi"]
    x = np.arange(len(rs))
    ax.plot(x, [100 * r["acc_complex"]["orbital"] for r in rs], "s-", color="#1f77b4",
            label="complex trunk + orbital")
    ax.plot(x, [100 * r["acc_complex"]["euclid"] for r in rs], "o-", color="#d62728",
            label="complex trunk + euclid")
    ax.plot(x, [100 * r["acc_real"]["euclid"] for r in rs], "^-", color="#7f7f7f",
            label="real trunk (2x params) + euclid")
    ax.set_xticks(x)
    ax.set_xticklabels([r["setting"] for r in rs], fontsize=7)
    ax.set_ylabel("acc (%) under inject pi")
    ax.set_title("F5b: pretrained-frozen trunks (3-way test classes)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F5_encoder.png"), dpi=160)
    print("F5 saved")


def f6_b7_budget():
    d = load("B7_endtoend.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    budgets = sorted(set(r["budget"] for r in d))
    for sig_key, title in [("sigma_3.14", "query inject sigma=pi"),
                           ("sigma_1.05", "query inject sigma=pi/3")]:
        for mode, col, mk in [("euclid", "#d62728", "o"), ("aug", "#9467bd", "s"),
                              ("orbital", "#2ca02c", "^")]:
            ys = [100 * np.mean([r["acc"][sig_key] for r in d
                                 if r["budget"] == b and r["mode"] == mode]) for b in budgets]
            ax.plot(budgets, ys, marker=mk, color=col,
                    label=f"{mode} ({title})", ls="-" if sig_key == "sigma_3.14" else "--")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("meta-training episodes")
    ax.set_ylabel("3-way test acc (%)")
    ax.set_title("F6: orbital objective beats augmentation at every meta-training budget")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F6_b7_budget.png"), dpi=160)
    print("F6 saved")


def t1_main_table():
    rows = load("B5_grid.json")
    if not rows:
        return
    methods = ["euclid", "cosine", "hermitian", "orbital", "tta_euclid", "circcoord",
               "whiten", "drop_agc", "phasemap", "phasemap_ml", "phasemap_w", "oracle"]
    lines = ["| sigma_th | shot | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 2)]
    for k in (5, 1):
        for r in sorted([r for r in rows if r["tag"] == "main" and r["p"] == 64
                         and r["rho"] == 0.3 and r["k"] == k and r["fade_q"] == 0.0],
                        key=lambda r: r["sigma_th"]):
            a = r["acc"]
            lines.append(f"| {r['sigma_th']:.2f} | {k} | " +
                         " | ".join(f"{100*a[m]:.1f}" for m in methods) + " |")
    open(os.path.join(TAB, "T1_synthetic_main.md"), "w", encoding="utf-8").write(
        "# T1 合成主网格（p=64, rho=0.3, fade=0, 2000 episodes/格, 5-way）\n\n"
        + "\n".join(lines) + "\n")
    print("T1 saved")


def t2_radioml_table():
    rows = load("B8_radioml.json")
    cross = load("B8_crosssnr.json")
    if not rows:
        return
    methods = ["euclid", "orbital", "tta_euclid", "circcoord", "whiten", "drop_agc",
               "phasemap", "phasemap_ml", "phasemap_w"]
    lines = ["| SNR | inject | shot | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 3)]
    for r in sorted(rows, key=lambda r: (r["snr"], r["inject_strength"], -r["k"])):
        inj = "natural" if r["inject_mode"] == "none" else f"sigma={r['inject_strength']:.2f}"
        lines.append(f"| {r['snr']}dB | {inj} | {r['k']} | " +
                     " | ".join(f"{100*r['acc'][m]:.1f}" for m in methods) + " |")
    if cross:
        avg = {m: float(np.mean([c["acc"][m] for c in cross])) for m in methods}
        lines.append("| 18->0 | cross-SNR | 5 | " +
                     " | ".join(f"{100*avg[m]:.1f}" for m in methods) + " |")
    open(os.path.join(TAB, "T2_radioml.md"), "w", encoding="utf-8").write(
        "# T2 真实IQ少样本（RML2016.10a, 11类5-way, 600 episodes/格, 能量归一化原始IQ特征）\n\n"
        + "\n".join(lines) + "\n")
    print("T2 saved")


def t3_h1b_tests():
    res = load("B5_extra_h1b_ml.json") or []
    lines = ["| rho | sigma_th | fade | phML | best endpoint | endpoint acc | mean diff | p | 95% CI |",
             "|---|---|---|---|---|---|---|---|---|"]
    for h in res:
        lines.append(
            f"| {h['rho']} | {h['sigma_th']:.2f} | {h['fade_q']} | "
            f"{100*h['acc_phasemap_ml']:.1f} | {h['better_endpoint']} | "
            f"{100*max(h['acc_euclid'], h['acc_orbital']):.1f} | {100*h['mean_d']:+.2f} | "
            f"{h['p']:.1e} | [{100*h['ci'][0]:+.2f},{100*h['ci'][1]:+.2f}] |")
    open(os.path.join(TAB, "T3_h1b_paired_tests.md"), "w", encoding="utf-8").write(
        "# T3 H1b paired tests (PhaseMAP-ML vs better-by-mean fixed endpoint, 2000 paired episodes)\n\n"
        + "\n".join(lines) + "\n")
    print("T3 saved")


def t4_b7_table():
    d = load("B7_endtoend.json")
    if not d:
        return
    lines = ["| budget | mode | sigma=0 | sigma=pi/3 | sigma=pi |",
             "|---|---|---|---|---|"]
    for b in sorted(set(r["budget"] for r in d)):
        for mode in ("euclid", "aug", "orbital"):
            rs = [r for r in d if r["budget"] == b and r["mode"] == mode]
            m = {k: 100 * np.mean([r["acc"][k] for r in rs])
                 for k in ("sigma_0.00", "sigma_1.05", "sigma_3.14")}
            lines.append(f"| {b} | {mode} | {m['sigma_0.00']:.1f} | "
                         f"{m['sigma_1.05']:.1f} | {m['sigma_3.14']:.1f} |")
    open(os.path.join(TAB, "T4_b7_endtoend.md"), "w", encoding="utf-8").write(
        "# T4 端到端原型训练（3-way测试类, 3种子均值, 等变复trunk）\n\n"
        + "\n".join(lines) + "\n")
    print("T4 saved")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    f1_main_curves()
    f2_adaptivity()
    f3_radioml()
    f4_theory()
    f5_b6_encoder()
    f6_b7_budget()
    t1_main_table()
    t2_radioml_table()
    t3_h1b_tests()
    t4_b7_table()
    print("B9 DONE")
