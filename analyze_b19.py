"""B19 分析（v2）：合并 full+fix 两轮，SC1–SC4 判定 + T10 表 + F9 图。

输入：04_results/logs/B19_{full,fix,synth,synth2}.json、B19_diag_local.json
输出：tables/T10_b19_equivariant.md、logs/B19_verdict.json、figures/F9_b19.png
"""
import json, os, sys
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LN5 = float(np.log(5))


def load(name):
    p = os.path.join(BASE, "04_results", "logs", name)
    return json.load(open(p)) if os.path.exists(p) else None


def acc(rec, key):
    return 100 * float(np.mean(rec["acc"][key]))


def main():
    full, fix = load("B19_full.json"), load("B19_fix.json")
    diag = load("B19_diag_local.json") or {}
    syn, syn2 = load("B19_synth.json"), load("B19_synth2.json")

    recs = {r["sigma"]: r for r in full["records"]}
    recf = {r["sigma"]: r for r in fix["records"]}
    sig_keys = sorted(recs.keys())
    sig0, sigP = sig_keys[0], sig_keys[-1]
    sig3 = min(sig_keys, key=lambda s: abs(s - np.pi / 3))

    arms = []
    for src, tag in ((full["train_hist"], ""), (fix["train_hist"], "")):
        for name in src:
            arms.append((name, src[name], full["meta"] if name in full["meta"] else fix["meta"]))
    seen, arm_list = set(), []
    for name, hist, meta in arms:
        if name not in seen:
            seen.add(name)
            arm_list.append((name, hist, meta))

    lines = ["# T10：B19 等变复 CNN 修复（自然 IQ，split0，budget=1200，3-way，chance=33.3，200 epi/格）", "",
             "| 臂 | 目标 | 参数 | loss末5 | δ等变 | 相消比 | Fisher | σ=0 (e/o/p) | σ=π/3 | σ=π (e/o/p) | 0→π轨道头降幅 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {"SC": {}, "arms": {}, "diag_local": diag}

    def get_rec(name, sg):
        return recf[sg] if f"{name}_euclid" in recf[sg]["acc"] or \
            any(k.startswith(name + "_") for k in recf[sg]["acc"]) else recs[sg]

    for name, hist, meta in arm_list:
        src_recs = recf if any(f"{name}_{h}" in recf[sig0]["acc"] for h in ("euclid", "orbital", "phML")) else recs
        l1 = float(np.mean(hist[-5:]))
        m = meta.get(name, {})
        heads = {}
        for sg in (sig0, sig3, sigP):
            hv = {hd: acc(src_recs[sg], f"{name}_{hd}")
                  for hd in ("euclid", "orbital", "phML") if f"{name}_{hd}" in src_recs[sg]["acc"]}
            heads[str(round(sg, 2))] = hv
        orb0 = heads[str(round(sig0, 2))].get("orbital")
        orbP = heads[str(round(sigP, 2))].get("orbital")
        drop = (orb0 - orbP) if (orb0 is not None and orbP is not None) else None
        obj = name.split("_")[-1]
        row = [name, obj, str(m.get("params", "")), f"{l1:.3f}",
               f"{m['equiv_delta']:.4f}" if "equiv_delta" in m else "—",
               f"{m['cancel_ratio']:.3f}" if "cancel_ratio" in m else "—",
               f"{m['fisher']:.2f}" if "fisher" in m else "—"]
        for sg in (sig0, sig3, sigP):
            hv = heads[str(round(sg, 2))]
            row.append("/".join(f"{hv.get(hd, float('nan')):.1f}"
                                for hd in ("euclid", "orbital", "phML")))
        row.append(f"{drop:+.1f}" if drop is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
        verdict["arms"][name] = {"loss_last5": l1, "heads": heads, "drop_0toPi_orb": drop,
                                 **{k: m.get(k) for k in ("equiv_delta", "cancel_ratio", "fisher")}}

    # ---- SC 判定（对严格等变 l1pca 臂 + power 折中臂）----
    real_best = max(max(v["heads"][str(round(sig0, 2))].values()) for n, v in verdict["arms"].items()
                    if n.startswith("real"))
    for nm in verdict["arms"]:
        v = verdict["arms"][nm]
        if not (nm.endswith("_l1pca") or nm == "v2_power_euclid"):
            continue
        best_nat = max(v["heads"][str(round(sig0, 2))].values())
        best_pi = max(v["heads"][str(round(sigP, 2))].values())
        verdict["SC"][nm] = {
            "SC1_train<=ln5-0.3": bool(v["loss_last5"] <= LN5 - 0.3),
            "SC2_acc>=real-3pt": bool(best_nat >= real_best - 3.0),
            "SC3_equiv_delta<=0.01": bool((v.get("equiv_delta") is not None)
                                          and v["equiv_delta"] <= 0.01),
            "SC4_drop0toPi<=1pt": bool(v["drop_0toPi_orb"] is not None
                                       and abs(v["drop_0toPi_orb"]) <= 1.0),
            "real_ref": real_best, "best_nat": best_nat,
        }

    # ---- 合成臂摘要 ----
    for tag, data in (("synth_D4_st0", syn), ("synth2_stpi", syn2)):
        if data:
            verdict[tag] = {k: {"first50": v["loss_first50"], "last50": v["loss_last50"]}
                            for k, v in data["meta"].items()}
            verdict[tag + "_eval"] = {c["arm"]: {k2: v2 for k2, v2 in c.items()
                                                 if k2.startswith(("acc_", "kappa"))}
                                      for c in data["records"]}

    out_t = os.path.join(BASE, "04_results", "tables", "T10_b19_equivariant.md")
    open(out_t, "w").write("\n".join(lines) + "\n")
    json.dump(verdict, open(os.path.join(BASE, "04_results", "logs", "B19_verdict.json"), "w"),
              indent=1)
    print("\n".join(lines))
    print(json.dumps(verdict["SC"], indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for name, hist, _ in arm_list:
            ls = "-" if ("l1pca" in name or "power" in name) else "--"
            axes[0].plot(np.arange(len(hist)) * 10, hist, ls, label=name, lw=1.2)
        axes[0].axhline(LN5, color="k", ls=":", lw=0.8)
        axes[0].text(5, LN5 + 0.01, "ln5", fontsize=7)
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("protonet loss")
        axes[0].set_title("B19 training (solid = phase-recovered/invariant, dashed = naive)")
        axes[0].legend(fontsize=6)
        names = [n for n, _, _ in arm_list]
        x = np.arange(len(names))
        for hd, mk in (("euclid", "o"), ("orbital", "s"), ("phML", "^")):
            vals = [acc(recs[sig0], f"{n}_{hd}") if f"{n}_{hd}" in recs[sig0]["acc"]
                    else (acc(recf[sig0], f"{n}_{hd}") if f"{n}_{hd}" in recf[sig0]["acc"]
                          else np.nan) for n in names]
            axes[1].plot(x, vals, marker=mk, ls="-", label=f"{hd} @σ=0")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=60, fontsize=6, ha="right")
        axes[1].set_ylabel("acc (%)"); axes[1].set_title("natural test acc by arm")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(BASE, "04_results", "figures", "F9_b19.png"), dpi=160)
        print("F9 saved")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
