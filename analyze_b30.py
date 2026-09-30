"""B30 分析：轨道池化 vs 门控池化的盆结构（PREREG_B30 判据 → T18/B30_verdict）。

读取三个臂各自目录的 B30_orbit_pool.json（分目录避免并发写竞态），合并后计算：
  SC-A-v5     orbit valsel 均值 ≥97 且好盆率(≥93) ≥80%
  SC-Orb-pair orbit−gated 好盆率差 ≥50pt（同 init×流配对）
  SC-Orb1     orbit init 间 sd ≤3pt
  SC-Eqv      每 run σ 配对平坦 ≤0.1pt 且 equiv_delta ≤1e-3
  SC-Hyb      hybrid 好盆率 ≥80%
  SC-Mech     orbit 的支持集对齐相干度 gamma_mag > gated + 0.3pt（配对）
输出 04_results/tables/T18_b30.md + logs/B30_verdict.json。
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, np.pi / 3, np.pi)


def load_arm(name):
    p = os.path.join(BASE, "04_results", "logs", f"B30_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {"runs": {}}


def acc(v, sel="valsel"):
    if sel not in v["eval"]:
        return None
    return 100 * float(np.mean([v["eval"][sel][f"sigma_{s:.2f}"]["orbital"] for s in SIG]))


def sigma_spread(v, sel="valsel"):
    if sel not in v["eval"]:
        return None
    a = [100 * v["eval"][sel][f"sigma_{s:.2f}"]["orbital"] for s in SIG]
    return max(a) - min(a)


def stats(runs, sel="valsel"):
    vs = np.array([acc(v) for v in runs.values() if acc(v, sel) is not None], float)
    good = vs >= 93
    return {"n": int(len(vs)), "mean": float(vs.mean()) if len(vs) else None,
            "sd": float(vs.std(ddof=1)) if len(vs) > 1 else None,
            "max": float(vs.max()) if len(vs) else None,
            "good_rate": float(good.mean()) if len(vs) else None,
            "good_n": int(good.sum()) if len(vs) else 0,
            "bimodal_gap": (float(min(vs[good]) - max(vs[~good])) if good.sum()
                            and (~good).sum() else None)}


def _gamma_mag(v):
    """eval[选点]['sigma_0.00']['gamma_mag']（支持集对齐相干度）。"""
    for sel in ("valsel", "endpoint", "ema"):
        gm = v["eval"].get(sel, {}).get("gamma_mag_0.00")
        if gm is not None:
            return gm
    return None


def paired_diff(runs_a, runs_b, sel="valsel"):
    # 两臂键前缀不同（orbit_i7 vs gated_i7）⇒ 按 init 匹配
    by_a, by_b = {}, {}
    for k, v in runs_a.items():
        by_a[k.split("_i")[-1]] = v
    for k, v in runs_b.items():
        by_b[k.split("_i")[-1]] = v
    keys = sorted(set(by_a) & set(by_b))
    d, gec = [], []
    for k in keys:
        a, b = acc(by_a[k], sel), acc(by_b[k], sel)
        if a is not None and b is not None:
            d.append(a - b)
            ga = _gamma_mag(by_a[k]); gb = _gamma_mag(by_b[k])
            if ga is not None and gb is not None:
                gec.append(100 * (ga - gb))
    p = float(wilcoxon(d).pvalue) if len(d) > 3 and not np.allclose(d, 0) else None
    return {"keys": [k.replace("orbit_", "").replace("hybrid_", "").replace("gated_", "")
                     for k in keys],
            "d": d, "mean_d": float(np.mean(d)) if d else None,
            "win_rate": float(np.mean([x > 0 for x in d])) if d else None,
            "p_wilcoxon": p, "gamma_mag_diff_pp": float(np.mean(gec)) if gec else None}


def main():
    extra = tuple(a for a in ("orbitL0.3", "learnlam")
                 if os.path.exists(os.path.join(LOG, f"B30_{a}.json")))
    arm_names = ("orbit", "hybrid", "gated") + extra
    arms = {a: load_arm(a)["runs"] for a in arm_names}
    out = {"n_runs": {a: len(r) for a, r in arms.items()},
           "stats": {a: {sel: stats(r, sel) for sel in ("valsel", "endpoint", "ema")}
                     for a, r in arms.items()},
           "paired_orbit_vs_gated": paired_diff(arms["orbit"], arms["gated"], "valsel"),
           "paired_hybrid_vs_gated": paired_diff(arms["hybrid"], arms["gated"], "valsel"),
           "equiv_delta_mean": {a: float(np.mean([v["info"]["equiv_delta"] for v in r.values()]))
                                for a, r in arms.items() if r},
           "sigma_spread_max": {a: max((sigma_spread(v) or 0) for v in r.values())
                                for a, r in arms.items() if r},
           "gamma_mag_mean": {
               a: float(np.mean([v["eval"].get("sigma_0.00", {}).get("gamma_mag", np.nan)
                                 for v in r.values()])) for a, r in arms.items() if r}}
    # 判据（全部由本脚本计算落盘）
    st_o = out["stats"]["orbit"]["valsel"]
    st_h = out["stats"]["hybrid"]["valsel"]
    st_g = out["stats"]["gated"]["valsel"]
    pd_ = out["paired_orbit_vs_gated"]
    verdict = {}
    if st_o["n"]:
        verdict["SC_A_v5_pass"] = bool(st_o["mean"] >= 97.0 and st_o["good_rate"] >= 0.8)
        verdict["SC_A_v5_mean"] = st_o["mean"]
        verdict["SC_A_v5_good_rate"] = st_o["good_rate"]
    if st_o["sd"] is not None:
        verdict["SC_Orb1_pass"] = bool(st_o["sd"] <= 3.0)
        verdict["SC_Orb1_sd"] = st_o["sd"]
    if st_g and st_g["good_rate"] is not None and st_o["good_rate"] is not None:
        verdict["SC_Orb_pair_pass"] = bool(st_o["good_rate"] - st_g["good_rate"] >= 0.5)
        verdict["SC_Orb_pair_gap"] = st_o["good_rate"] - st_g["good_rate"]
    if out["sigma_spread_max"].get("orbit") is not None:
        verdict["SC_Eqv_pass"] = bool(out["sigma_spread_max"]["orbit"] <= 0.1
                                      and out["equiv_delta_mean"]["orbit"] <= 1e-3)
        verdict["SC_Eqv_sigma_spread_max"] = out["sigma_spread_max"]["orbit"]
        verdict["SC_Eqv_delta"] = out["equiv_delta_mean"]["orbit"]
    if st_h and st_h["n"]:
        verdict["SC_Hyb_pass"] = bool(st_h["good_rate"] >= 0.8)
        verdict["SC_Hyb_good_rate"] = st_h["good_rate"]
        verdict["SC_Hyb_mean"] = st_h["mean"]
    if pd_["gamma_mag_diff_pp"] is not None:
        verdict["SC_Mech_pass"] = bool(pd_["gamma_mag_diff_pp"] > 0.3)
        verdict["SC_Mech_gamma_mag_diff_pp"] = pd_["gamma_mag_diff_pp"]
    out["verdict"] = verdict

    os.makedirs(os.path.join(BASE, "04_results", "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(LOG, "B30_verdict.json"), "w"), indent=1)

    tab = ["# T18 — B30 轨道池化（分析式相位同步）vs 门控池化（PREREG_B30）", "",
           "5 臂（gated 对照 / orbit λ=1 / hybrid / orbit λ=0.3 / learnlam）× 16 init"
           "（与 B29d 同 init×流 4000）× 3200 步；评估 σ 配对（同 base episodes，"
           "平坦性=精确等变证书）。", "",
           "| 臂 | n | valsel 均值 | sd | max | 好盆率(≥93) | 双盆 gap |", "|---|---|---|---|---|---|---|"]
    for a in arm_names:
        s = out["stats"].get(a, {}).get("valsel", {"n": 0})
        if s["n"]:
            tab.append(f"| {a} | {s['n']} | {s['mean']:.2f} | {s['sd']:.2f} | {s['max']:.1f} "
                       f"| {s['good_rate']:.0%} ({s['good_n']}) | "
                       f"{s['bimodal_gap'] if s['bimodal_gap'] is not None else '—'} |")
    pd1 = out["paired_orbit_vs_gated"]
    tab += ["", f"配对 orbit−gated（同 init×流，n={len(pd1['d'])}）：均值差 "
           f"{pd1['mean_d']:+.2f}pt，orbit 胜率 {pd1['win_rate']:.0%}，"
           f"Wilcoxon p={pd1['p_wilcoxon']:.2e}" if pd1["mean_d"] is not None else "",
           f"机制（支持集对齐相干度 gamma_mag）：orbit−gated = "
           f"{pd1['gamma_mag_diff_pp']:+.2f}pp（配对均值）"
           if pd1["gamma_mag_diff_pp"] is not None else "",
           "", "## 判据（PREREG_B30，运行前写定）", ""]
    for k, v in verdict.items():
        tab.append(f"- {k}: {v}")
    tab += ["", "## 逐 run（valsel 三头均值）", "",
            "| init | orbit | hybrid | gated |", "|---|---|---|---|"]
    for k in pd1["keys"]:
        r = [acc(arms[a].get(f"{a}_{k}")) if arms[a].get(f"{a}_{k}") else None
             for a in arm_names]
        tab.append("| " + " | ".join(f"{x:.1f}" if x is not None else "—" for x in r) + " |")
    with open(os.path.join(BASE, "04_results", "tables", "T18_b30.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tab))
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"-> {LOG}/B30_verdict.json, T18_b30.md")


if __name__ == "__main__":
    main()
