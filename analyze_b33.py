"""B33 分析：第二类划分（split-B: 8PSK/CPFSK/QAM64）上的复合不变性复现验证。

目的（回应审稿 B-05"3 测试类窄 scope"）：B32 结论（invpow_metric 16/16 好盆、
构造性不变、sd 全项目最低）在与 split0 零重叠的测试类划分上是否复现。
协议与 B32 完全相同（同 init×流、3200 步、σ 配对注入、disjoint），仅改类划分。
对照臂：gated 等变学习臂基线。判据沿用 PREREG_B32（SC-A/SC-Rel/SC-Pair/SC-Cert），
在 n=8 下评估（统计功效如实标注）。
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, np.pi / 3, np.pi)
ARM_HEAD = {"invpow_metric": "euclid", "metric": "euclid", "invpow": "euclid",
            "autocorr": "euclid", "power": "euclid", "nopool": "orbital",
            "gated": "orbital"}


def load_arm(name):
    p = os.path.join(LOG, f"B33_{name}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        runs = d["runs"]
        meta = d.get("meta", {})
        return runs, meta
    return {}, {}


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
    arms = ["invpow_metric"]                  # 主臂；gated 为基线对照
    runs, metas = {}, {}
    for a in arms + ["gated"]:
        runs[a], metas[a] = load_arm(a)
    heads = {a: ARM_HEAD[a] for a in runs}
    st = {a: stats(runs[a], heads[a]) for a in runs}

    out = {"purpose": "第二类划分（split-B）上 B32 结论的复现验证（审稿 B-05）",
           "split": metas.get("invpow_metric", {}), "heads": heads, "stats": st,
           "paired_vs_gated": {a: paired(runs[a], runs["gated"], heads[a], heads["gated"])
                               for a in arms}}

    verdict = {}
    a = "invpow_metric"
    if st[a].get("n", 0) >= 3:
        sb = st[a]
        verdict["n_init"] = sb["n"]
        verdict["SC_A_pass_splitB"] = bool(sb["mean"] >= 93 and sb["good_rate"] >= 0.5)
        verdict["mean_splitB"] = sb["mean"]
        verdict["good_rate_splitB"] = sb["good_rate"]
        verdict["sd_splitB"] = sb.get("sd")
        pdg = out["paired_vs_gated"].get(a)
        verdict["SC_Pair_pass_splitB"] = bool(pdg and pdg["mean_d"] >= 5.0
                                              and pdg["p"] is not None and pdg["p"] < 0.05)
        verdict["SC_Pair_splitB"] = pdg
        verdict["SC_Cert_pass_splitB"] = bool(sb.get("invariance_delta_max", 9) <= 1e-3
                                               and sb.get("sigma_flat_max", 9) <= 0.1)
        verdict["SC_Cert_invariance_delta_max"] = sb.get("invariance_delta_max")
        verdict["SC_Cert_sigma_flat_max"] = sb.get("sigma_flat_max")
        # B32 复现判据：好盆率在 split-B 上不低于 50%（n=8 下功效有限，如实标注）
        verdict["reproduce_B32_good_rate"] = bool(sb["good_rate"] >= 0.5)
        verdict["reproduce_B32_narrow_sd"] = bool(sb.get("sd", 99) <= 6)
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(LOG, "B33_verdict.json"), "w"), indent=1)

    sb_meta = metas.get("invpow_metric", {})
    tab = ["# T21 — B33 第二类划分复现验证（split-B: 8PSK/CPFSK/QAM64）", "",
           f"目的：B32 复合不变性结论在与 split0（AM-DSB/QAM16/QPSK）零测试类重叠的新划分上"
           "是否复现（审稿 B-05 窄 scope）。", "",
           f"协议同 B32（同 init×流逐 run 配对、3200 步、σ 配对注入、disjoint）；"
           f"n={sb_meta.get('n_inits','?')} 个 init（B32 主实验为 16，本验证为 8，统计功效较低如实标注）。",
           f"测试类：{sb_meta.get('split_test_classes','?')}", "",
           "| 臂 | n | valsel 均值 | sd | max | min | 好盆(≥93) | σ 平坦 | δ_inv |",
           "|---|---|---|---|---|---|---|---|---|"]
    for arm_name in list(runs.keys()):
        s = st[arm_name]
        if s.get("n"):
            tab.append(f"| {arm_name} | {s['n']} | {s['mean']:.2f} | {s['sd']:.2f} "
                       f"| {s['max']:.1f} | {s['min']:.1f} | {s['good_rate']:.0%} "
                       f"({s['good_n']}) | {s.get('sigma_flat_max', float('nan')):.3f} | "
                       f"{s.get('invariance_delta_max', float('nan')):.1e} |")
    tab += ["", "## 配对差（同 init×流）", ""]
    for arm_name in arms:
        pd_ = out["paired_vs_gated"].get(arm_name)
        if pd_:
            tab.append(f"- {arm_name} − gated: {pd_['mean_d']:+.2f}pt"
                       f"（胜率 {pd_['win_rate']:.0%}，Wilcoxon p={pd_['p']}, n={pd_['n']}）")
    tab += ["", "## 判据（沿用 PREREG_B32，split-B 上重估；n=8 功效标注）", ""]
    for k, v in verdict.items():
        if not isinstance(v, dict):
            tab.append(f"- {k}: {v}")
    with open(os.path.join(BASE, "04_results", "tables", "T21_b33.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tab))
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print("-> B33_verdict.json, T21_b33.md")


if __name__ == "__main__":
    main()
