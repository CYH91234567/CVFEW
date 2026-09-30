"""analyze_b25：SC-A 第三次攻坚判定（T13 + B25_verdict.json）。

输入：04_results/logs/B25_sca.json（run_b25.py 输出）。
判定（PREREG_B25）：
  SC-A  最优臂三种子轨道头均值 ≥ 97.0（=real@3200 −2pt）
  SC-B  该臂 σ_θ 极差 ≤1pt 且 δ ≤0.01
  SC-M1 es3200 > base1200（早停修复预算轴）
  SC-M2 soft1200 训练起来（loss_last100 < ln5−0.3）且 > base1200
  SC-M3 big1200 > base1200（容量在好预算区有效）
  SC-M4 l1pca2 vs base1200 方向记录
另做：跨种子方差表（B22 容量归因修正的依据）、es 的 val-test 错位量化。
"""
import json, os, sys
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_results", "logs")
SIG = ("sigma_0.00", "sigma_1.05", "sigma_3.14")
ARMS = ["base1200", "es3200", "big1200", "big_es3200", "soft1200", "l1pca2_1200"]


def arm_stat(runs, arm, field="eval"):
    """返回 dict seed → (orbital_mean, flatness, per-sigma orbital)。"""
    out = {}
    for s in (11, 23, 37):
        key = f"{arm}_s{s}"
        if key not in runs:
            continue
        ev = runs[key][field]
        orb = [100 * ev[g]["orbital"] for g in SIG]
        out[s] = {"orbital_mean": float(np.mean(orb)),
                  "orbital_flat": max(orb) - min(orb),
                  "orbital": orb,
                  "phml_mean": float(np.mean([100 * ev[g]["phML"] for g in SIG])),
                  "euclid_mean": float(np.mean([100 * ev[g]["euclid"] for g in SIG]))}
    return out


def main():
    runs = json.load(open(os.path.join(LOG, "B25_sca.json")))["runs"]
    lines = ["# T13：B25——SC-A 第三次攻坚（自然 IQ split0，3-way，orbital 头，3 种子）", "",
             "| 臂 | s11 | s23 | s37 | 均值 | sd | 极差 | δ(均值) | loss末 |", "|---|---|---|---|---|---|---|---|---|"]
    stat = {}
    for arm in ARMS:
        st = arm_stat(runs, arm)
        if not st:
            continue
        stat[arm] = st
        means = [v["orbital_mean"] for v in st.values()]
        deltas = [runs[f"{arm}_s{s}"]["info"].get("equiv_delta", np.nan) for s in st]
        losses = [runs[f"{arm}_s{s}"]["info"].get("loss_last100", np.nan) for s in st]
        flats = [v["orbital_flat"] for v in st.values()]
        lines.append("| " + arm + " | " + " | ".join(f"{m:.1f}" for m in means) +
                     f" | {np.mean(means):.1f} | {np.std(means, ddof=1):.1f} |" +
                     f" {np.mean(flats):.1f} | {np.nanmean(deltas)*1:.4f} | {np.mean(losses):.3f} |")
        for s, v in st.items():
            lines.append(f"|  └ s{s} | | | | | |" +
                         f" σ: {'/'.join(f'{x:.1f}' for x in v['orbital'])} | | |")
    # EMA/EMA 列（B25b 除外，此处 es 臂的 best_val 记录）
    for arm in ("es3200", "big_es3200"):
        for s in (11, 23, 37):
            key = f"{arm}_s{s}"
            if key in runs:
                info = runs[key]["info"]
                lines.append(f"- {key}: best_val={info.get('best_val', float('nan')):.3f}"
                             f"@it{info.get('best_it', -1)}, stopped@{info.get('stopped_at')}")
    txt = "\n".join(lines)

    # 判定
    def m3(arm, field="orbital_mean"):
        return (np.mean([stat[arm][s][field] for s in stat[arm]])
                if arm in stat and len(stat[arm]) == 3 else np.nan)

    verdict = {"real3200_ref": 99.0, "threshold_SC_A": 97.0}
    cand = {a: m3(a) for a in stat}
    best_arm = max(cand, key=cand.get) if cand else None
    verdict["arm_means"] = cand
    verdict["best_arm"] = best_arm
    if best_arm:
        best_flat = np.mean([stat[best_arm][s]["orbital_flat"] for s in stat[best_arm]])
        best_delta = np.nanmean([runs[f"{best_arm}_s{s}"]["info"].get("equiv_delta", np.nan)
                                 for s in stat[best_arm]])
        verdict["SC_A_pass"] = bool(cand[best_arm] >= 97.0)
        verdict["SC_B"] = {"flat": float(best_flat), "delta": float(best_delta),
                           "pass": bool(best_flat <= 1.0 and best_delta <= 0.01)}
    verdict["SC_M1_es_fixes_budget"] = bool(m3("es3200") > m3("base1200"))
    ln5 = float(np.log(5))
    soft_trained = all(runs[f"soft1200_s{s}"]["info"]["loss_last100"] < ln5 - 0.3
                       for s in stat.get("soft1200", {})) if "soft1200" in stat else False
    verdict["SC_M2_soft"] = {"trains": bool(soft_trained),
                             "better": bool(m3("soft1200") > m3("base1200"))}
    verdict["SC_M3_big_at_1200"] = bool(m3("big1200") > m3("base1200"))
    verdict["SC_M4_l1pca2_vs_base"] = float(m3("l1pca2_1200") - m3("base1200"))
    # 跨种子方差（归因修正）
    verdict["seed_sd"] = {a: float(np.std([v["orbital_mean"] for v in st.values()], ddof=1))
                          for a, st in stat.items()}
    # es val-test 错位
    mis = []
    for arm in ("es3200", "big_es3200"):
        for s in stat.get(arm, {}):
            key = f"{arm}_s{s}"
            bv = runs[key]["info"].get("best_val")
            if bv is not None:
                mis.append({"arm": key, "best_val": 100 * bv,
                            "test_orbital_mean": stat[arm][s]["orbital_mean"],
                            "base_same_seed": stat.get("base1200", {}).get(s, {}).get("orbital_mean")})
    verdict["es_val_test"] = mis

    out_md = os.path.join(LOG, "..", "tables", "T13_b25_sca.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(txt + "\n\n## 判定\n```json\n" + json.dumps(verdict, indent=1, ensure_ascii=False) + "\n```\n")
    json.dump(verdict, open(os.path.join(LOG, "B25_verdict.json"), "w"), indent=1)
    print(txt)
    print(json.dumps(verdict, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
