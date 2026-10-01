"""B31 分析：不变结构读出（autocorr/power 不变臂用 euclid 头；nopool 用 orbital 头）
对照 B30 gated（同 init×流 4000、同 budget、同 σ 配对评估——逐 run 配对，不重跑）。
判据见 PREREG_B31 → B31_verdict.json + T19 表。
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, np.pi / 3, np.pi)
# 每臂的主评估头：不变臂嵌入无全局相位（对齐=破坏信息，B30 教训）⇒ euclid；
# nopool 是等变复嵌入，全局相位由原型对齐处理 ⇒ orbital。
ARM_HEAD = {"autocorr": "euclid", "power": "euclid", "nopool": "orbital",
            "gated": "orbital"}


def load_arm(name):
    p = os.path.join(LOG, f"B30_{name}.json")
    return json.load(open(p))["runs"] if os.path.exists(p) else {}


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
    return {"n": int(len(vs)), "mean": float(vs.mean()), "sd": float(vs.std(ddof=1)),
            "max": float(vs.max()), "good_rate": float(good.mean()),
            "good_n": int(good.sum())}


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
    arms = ["autocorr", "nopool", "power", "gated"]
    runs = {a: load_arm(a) for a in arms}
    heads = {a: ARM_HEAD.get(a, "orbital") for a in arms}
    out = {"heads": heads,
           "stats": {a: stats(runs[a], heads[a]) for a in arms},
           "paired_vs_gated": {a: paired(runs[a], runs["gated"], heads[a], heads["gated"])
                               for a in ("autocorr", "nopool", "power")}}
    # SC-NoC：autocorr 精度与隐层相干 cancel_ratio 的相关性（不变读出应免疫）
    ac = runs["autocorr"]
    pairs = [(acc(v, "euclid"), v["info"].get("cancel_ratio")) for v in ac.values()]
    pairs = [(a, c) for a, c in pairs if a is not None and c is not None]
    if len(pairs) >= 5:
        rho, rho_p = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        out["SC_NoCo_spearman"] = {"rho": float(rho), "p": float(rho_p), "n": len(pairs)}
    # SC-Flat：每臂每 run 的 σ 平坦（配对 base episodes ⇒ 构造性 ≤0.1pt+float）
    flat = {}
    for a in arms:
        sp = []
        for v in runs[a].values():
            e = v["eval"].get("valsel")
            if e:
                vals = [100 * e[f"sigma_{s:.2f}"][heads[a]] for s in SIG]
                sp.append(max(vals) - min(vals))
        if sp:
            flat[a] = {"max": float(np.max(sp)), "mean": float(np.mean(sp))}
    out["sigma_flat_pt"] = flat

    st = out["stats"]
    verdict = {}
    sa = st["autocorr"]
    if sa.get("n"):
        verdict["SC_Inv_a_pass"] = bool(sa["mean"] >= 90)
        verdict["SC_Inv_a_mean"] = sa["mean"]
        verdict["SC_Inv_b_pass"] = bool(sa["sd"] <= 3 and sa["good_rate"] >= 0.8)
        verdict["SC_Inv_b_sd"] = sa["sd"]; verdict["SC_Inv_b_good_rate"] = sa["good_rate"]
    if "SC_NoCo_spearman" in out:
        verdict["SC_NoCo_pass"] = bool(abs(out["SC_NoCo_spearman"]["rho"]) <= 0.3)
        verdict["SC_NoCo_rho"] = out["SC_NoCo_spearman"]["rho"]
    if flat.get("autocorr"):
        verdict["SC_Flat_pass"] = bool(flat["autocorr"]["max"] <= 0.5)
        verdict["SC_Flat_max_pt"] = flat["autocorr"]["max"]
    sn = st["nopool"]
    if sn.get("n"):
        verdict["SC_Nopool_pass"] = bool(sn["mean"] >= 85)
        verdict["SC_Nopool_mean"] = sn["mean"]
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(LOG, "B31_verdict.json"), "w"), indent=1)

    tab = ["# T19 — B31 不变结构读出（PREREG_B31；对照 B30 gated，同 init×流配对）", "",
           "autocorr = {R(τ) τ=1..6, m2, m4}（C×8，精确全局相位不变+结构保持）；"
           "nopool = 隐层序列拉平（等变，l1pca 原型）；power = 幅度不变量基线。", "",
           "| 臂 | 评估头 | n | valsel 均值 | sd | max | 好盆(≥93) | σ 平坦 max(pt) |",
           "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = st[a]
        if s.get("n"):
            tab.append(f"| {a} | {heads[a]} | {s['n']} | {s['mean']:.2f} | {s['sd']:.2f} "
                       f"| {s['max']:.1f} | {s['good_rate']:.0%} ({s['good_n']}) | "
                       f"{flat.get(a, {}).get('max', float('nan')):.2f} |")
    tab += ["", "## 配对差（arm − gated，同 init×流）", ""]
    for a in ("autocorr", "nopool", "power"):
        pd_ = out["paired_vs_gated"].get(a)
        if pd_:
            tab.append(f"- {a}: {pd_['mean_d']:+.2f}pt（胜率 {pd_['win_rate']:.0%}，"
                       f"Wilcoxon p={pd_['p']:.2e}，n={pd_['n']})")
    tab += ["", "## 判据（PREREG_B31）", ""]
    for k, v in verdict.items():
        tab.append(f"- {k}: {v}")
    with open(os.path.join(BASE, "04_results", "tables", "T19_b31.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tab))
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"-> B31_verdict.json, T19_b31.md")


if __name__ == "__main__":
    main()
