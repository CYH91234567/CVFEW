"""B32 分析：复合不变性（不变读出+学习度量头）→ B32_verdict.json / T20_b32.md。

对照两套配对：(a) vs B30 gated（等变学习臂基线）；(b) vs B31 autocorr
（无度量头的不变基线）。归因分解：invpow−autocorr = H-cap；metric−autocorr =
H-metric。判据见 PREREG_B32（SC-A-v6 / SC-Rel / SC-Pair / SC-Cert）。
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, np.pi / 3, np.pi)
ARM_HEAD = {"metric": "euclid", "invpow": "euclid", "invpow_metric": "euclid",
            "autocorr": "euclid", "power": "euclid", "nopool": "orbital",
            "gated": "orbital"}


def load_arm(name, b32=True):
    for fn in ([f"B32_{name}.json"] if b32 else []) + [f"B30_{name}.json"]:
        p = os.path.join(LOG, fn)
        if os.path.exists(p):
            return json.load(open(p))["runs"]
    return {}


def acc(v, head):
    e = v["eval"].get("valsel")
    if not e:
        return None
    return 100 * float(np.mean([e[f"sigma_{s:.2f}"][head] for s in SIG]))


def stats(runs, head):
    vs = np.array([acc(v, head) for v in runs.values() if acc(v, head) is not None])
    if not len(vs):
        return {"n": 0}
    good = vs >= 93
    st = {"n": int(len(vs)), "mean": float(vs.mean()), "sd": float(vs.std(ddof=1)),
          "max": float(vs.max()), "min": float(vs.min()),
          "good_rate": float(good.mean()), "good_n": int(good.sum())}
    if good.sum() and (~good).sum():
        st["bimodal"] = {"good_center": float(vs[good].mean()),
                         "bad_center": float(vs[~good].mean()),
                         "gap": float(vs[good].min() - vs[~good].max()),
                         "n_good": int(good.sum()), "n_bad": int((~good).sum())}
    invid = [v["info"].get("invariance_delta") for v in runs.values()
             if v["info"].get("invariance_delta") is not None]
    if invid:
        st["invariance_delta_max"] = float(np.max(invid))
    fl = [max(100 * e2[f"sigma_{s:.2f}"][head] for s in SIG)
          - min(100 * e2[f"sigma_{s:.2f}"][head] for s in SIG)
          for e2 in (v["eval"].get("valsel") for v in runs.values()) if e2]
    if fl:
        st["sigma_flat_max"] = float(np.max(fl))
    return st


def paired(runs_a, runs_b, head_a, head_b):
    by_a = {k.split("_i")[-1]: v for k, v in runs_a.items()}
    by_b = {k.split("_i")[-1]: v for k, v in runs_b.items()}
    keys = sorted(set(by_a) & set(by_b))
    d = [(acc(by_a[k], head_a), acc(by_b[k], head_b)) for k in keys]
    d = [(a, b) for a, b in d if a is not None and b is not None]
    if not d:
        return None
    da = [a - b for a, b in d]
    p = float(wilcoxon(da).pvalue) if len(da) > 3 and not np.allclose(da, 0) else None
    return {"n": len(da), "mean_d": float(np.mean(da)),
            "win_rate": float(np.mean([x > 0 for x in da])), "p": p}


def main():
    arms = ["metric", "invpow", "invpow_metric"]
    runs = {a: load_arm(a) for a in arms}
    runs["autocorr"] = load_arm("autocorr", b32=False)     # B31 基线
    runs["gated"] = load_arm("gated", b32=False)           # B29d/B30 对照
    heads = {a: ARM_HEAD.get(a, "euclid") for a in runs}
    st = {a: stats(runs[a], heads[a]) for a in runs}

    out = {"heads": heads, "stats": st,
           "paired_vs_gated": {a: paired(runs[a], runs["gated"], heads[a], heads["gated"])
                               for a in arms},
           "paired_vs_autocorr": {a: paired(runs[a], runs["autocorr"], heads[a],
                                            heads["autocorr"]) for a in arms}}

    verdict = {}
    best = max((a for a in arms if st[a].get("n", 0) >= 5), key=lambda a: st[a]["mean"],
               default=None)
    if best:
        sb = st[best]
        verdict["SC_A_v6_pass"] = bool(sb["mean"] >= 93 and sb["good_rate"] >= 0.5)
        verdict["SC_A_v6_arm"] = best
        verdict["SC_A_v6_mean"] = sb["mean"]
        verdict["SC_A_v6_good_rate"] = sb["good_rate"]
        verdict["SC_Rel_pass"] = bool(sb["mean"] >= 88 and sb.get("sd", 99) <= 6)
        verdict["SC_Rel_mean"] = sb["mean"]
        verdict["SC_Rel_sd"] = sb.get("sd")
        pdg = out["paired_vs_gated"].get(best)
        verdict["SC_Pair_pass"] = bool(pdg and pdg["mean_d"] >= 5.0 and pdg["p"] is not None
                                       and pdg["p"] < 0.05)
        verdict["SC_Pair"] = pdg
        verdict["SC_Cert_pass"] = bool(sb.get("invariance_delta_max", 9) <= 1e-3
                                       and sb.get("sigma_flat_max", 9) <= 0.1)
        verdict["SC_Cert_invariance_delta_max"] = sb.get("invariance_delta_max")
        verdict["SC_Cert_sigma_flat_max"] = sb.get("sigma_flat_max")
    for a in arms:                                       # 归因分解（H-cap / H-metric）
        pda = out["paired_vs_autocorr"].get(a)
        if pda and st["autocorr"].get("n"):
            out[f"decomp_{a}"] = {"d_vs_autocorr_pt": pda["mean_d"], "p": pda["p"],
                                  "n": pda["n"],
                                  "hypothesis": ("H-cap" if a == "invpow"
                                                 else "H-metric" if a == "metric"
                                                 else "H-cap+H-metric")}
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(LOG, "B32_verdict.json"), "w"), indent=1)

    tab = ["# T20 — B32 复合不变性：不变读出 + 学习度量头（PREREG_B32）", "",
           "16 init × 3200 步，与 B29d/B30/B31 同 init×流逐 run 配对；不变臂 euclid 头。",
           "对照：B30 gated（等变学习臂）与 B31 autocorr（无度量头不变基线）。", "",
           "| 臂 | n | valsel 均值 | sd | max | min | 好盆(≥93) | σ 平坦 | δ_inv |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in list(runs.keys()):
        s = st[a]
        if s.get("n"):
            tab.append(f"| {a} | {s['n']} | {s['mean']:.2f} | {s['sd']:.2f} | {s['max']:.1f} "
                       f"| {s['min']:.1f} | {s['good_rate']:.0%} ({s['good_n']}) | "
                       f"{s.get('sigma_flat_max', float('nan')):.3f} | "
                       f"{s.get('invariance_delta_max', float('nan')):.1e} |")
    tab += ["", "## 配对差（同 init×流）", ""]
    for a in arms:
        for ref, key in (("gated", "paired_vs_gated"), ("autocorr", "paired_vs_autocorr")):
            pd_ = out[key].get(a)
            if pd_:
                tab.append(f"- {a} − {ref}: {pd_['mean_d']:+.2f}pt（胜率 {pd_['win_rate']:.0%}，"
                           f"Wilcoxon p={pd_['p']}, n={pd_['n']}）")
    tab += ["", "## 判据（PREREG_B32，运行前写定）", ""]
    for k, v in verdict.items():
        if not isinstance(v, dict):
            tab.append(f"- {k}: {v}")
    with open(os.path.join(BASE, "04_results", "tables", "T20_b32.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tab))
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"-> B32_verdict.json, T20_b32.md")


if __name__ == "__main__":
    main()
