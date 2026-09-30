"""analyze_b27：B7 复访判定（T14 + B27_verdict.json）。

输入：04_results/logs/B27_b7revisit.json（orbital / orbital_l1pca 重跑）
     + 04_results/logs/B7_endtoend.json（B7 原始 euclid/aug/orbital，同种子同流）。
判定（PREREG_B27）：
  SC-A  orbital_l1pca 全部 3 预算 × 3 种子的 σ_θ=π 评估均值 > aug 同格均值
  SC-B  orbital_l1pca ≥ orbital（朴素）全格
  SC-C  l1pca 臂 σ_θ=0→π 降幅 ≤ aug 臂
主指标 = B7 原判据（均值原型欧氏 d2）。
"""
import json, os
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_results", "logs")
BUDGETS = (200, 800, 3200)
SEEDS = (0, 1, 2)


def main():
    b27 = json.load(open(os.path.join(LOG, "B27_b7revisit.json")))["runs"]
    b7 = json.load(open(os.path.join(LOG, "B7_endtoend.json")))
    # B7 原始键为 sigma_*（其评估即均值原型欧氏），统一改名为 meanproto_*
    b7map = {(r["budget"], r["mode"], r["seed"]):
             {f"meanproto_{k.split('_')[1]}": v for k, v in r["acc"].items()} for r in b7}

    def key_of(budget, mode, seed, prefix="meanproto_"):
        for r in b27:
            if r["budget"] == budget and r["mode"] == mode and r["seed"] == seed:
                return r["acc"]
        return b7map.get((budget, mode, seed))

    def acc_at(acc, ev, prefix="meanproto_"):
        if acc is None:
            return np.nan
        kk = [k for k in acc if k.startswith(prefix) and abs(float(k[len(prefix):]) - ev) < 5e-3]
        return 100 * acc[kk[0]] if kk else np.nan

    lines = ["# T14：B27——B7 复访（l1pca 原型目标 vs 增广；ComplexTrunk，自然 IQ split0，3 种子）", "",
             "| 预算 | 臂 | σ=0 | π/3 | π | σ=0→π 降幅 |", "|---|---|---|---|---|---|"]
    grid = {}
    for b in BUDGETS:
        for mode in ("euclid", "aug", "orbital", "orbital_l1pca"):
            vals = {ev: [acc_at(key_of(b, mode, s), ev) for s in SEEDS]
                    for ev in (0.0, np.pi / 3, np.pi)}
            if all(np.isnan(v).all() for v in vals.values()):
                continue
            m = {ev: np.nanmean(vals[ev]) for ev in vals}
            drop = m[0.0] - m[np.pi]
            grid[(b, mode)] = {"mean": m, "drop": drop,
                               "sd_pi": float(np.nanstd(vals[np.pi], ddof=1))}
            lines.append(f"| {b} | {mode} | {m[0.0]:.1f} | {m[np.pi/3]:.1f} | {m[np.pi]:.1f} | {drop:+.1f} |")

    verdict = {}
    scA, scB, scC = True, True, True
    for b in BUDGETS:
        for s_i, s in enumerate(SEEDS):
            lp = acc_at(key_of(b, "orbital_l1pca", s), np.pi)
            aug = acc_at(key_of(b, "aug", s), np.pi)
            orb = acc_at(key_of(b, "orbital", s), np.pi)
            if not (lp > aug):
                scA = False
            if not (lp >= orb):
                scB = False
        d_lp = grid[(b, "orbital_l1pca")]["drop"]
        d_aug = grid[(b, "aug")]["drop"]
        if not (d_lp <= d_aug + 1e-9):
            scC = False
    verdict["SC_A_l1pca_gt_aug_all_cells"] = bool(scA)
    verdict["SC_B_l1pca_ge_orbital_all_cells"] = bool(scB)
    verdict["SC_C_flatness_vs_aug"] = bool(scC)
    verdict["margin_3200"] = {
        "l1pca_pi": grid[("3200" if False else 3200, "orbital_l1pca")]["mean"][np.pi],
        "aug_pi": grid[(3200, "aug")]["mean"][np.pi],
        "orbital_pi": grid[(3200, "orbital")]["mean"][np.pi]}
    verdict["grid"] = {f"{b}_{m}": v for (b, m), v in grid.items()}

    out_md = os.path.join(LOG, "..", "tables", "T14_b27_b7revisit.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n\n## 判定\n```json\n" +
                json.dumps(verdict, indent=1, ensure_ascii=False) + "\n```\n")
    json.dump(verdict, open(os.path.join(LOG, "B27_verdict.json"), "w"), indent=1)
    print("\n".join(lines))
    print(json.dumps(verdict, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
