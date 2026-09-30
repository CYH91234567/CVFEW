"""B29d 分析：早期盆筛选判定（PREREG_B29d）→ T17/B29d_verdict.json。

 SC-S1  cancel_ratio@800 对好盆（valsel≥93）的 AUC ≥ 0.8（Mann-Whitney）
 SC-S2  cancel_ratio@800 中位数上半组的好盆率 ≥ 60%（基线 33%）
 SC-S3  vt@800 的 AUC ≤ 0.6（vt 无预测力的稳健性确认）
 SC-A-v4 16 runs valsel 均值 + 双盆结构复现
 附带：cancel/vt 各时点 AUC 曲线（论文图 F13 候选）
"""
import json, os
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
res = json.load(open(os.path.join(BASE, "04_results", "logs", "B29d_screening.json")))
runs = res["runs"]
SIG = (0.0, np.pi / 3, np.pi)


def hm(v, w="valsel"):
    if w not in v["eval"]:
        return None
    return 100 * float(np.mean([v["eval"][w][f"sigma_{s:.2f}"]["orbital"] for s in SIG]))


keys = sorted(runs)
vs = np.array([hm(v) for v in runs.values()])
ep = np.array([hm(v, "endpoint") for v in runs.values()])
good = vs >= 93


def auc(scores):
    s = np.asarray(scores, float)
    if good.sum() == 0 or good.sum() == len(good):
        return None
    return mannwhitneyu(s[good], s[~good]).statistic / (good.sum() * (~good).sum())


its = [100, 200, 400, 600, 800, 1000, 1600, 2400, 3200]
cr_by_it = {it: [dict((d["it"], d) for d in v["info"]["diag"]).get(it, {}).get("cancel")
                 for v in runs.values()] for it in its}
vt_by_it = {it: [dict((d["it"], d) for d in v["info"]["diag"]).get(it, {}).get("vt")
                 for v in runs.values()] for it in its}
cr_auc = {it: auc(cr_by_it[it]) for it in its if all(x is not None for x in cr_by_it[it])}
vt_auc = {it: auc(vt_by_it[it]) for it in its if all(x is not None for x in vt_by_it[it])}
cr_end = [v["info"]["cancel_endpoint"] for v in runs.values()]

out = {"n_runs": len(runs), "good_basin_n": int(good.sum()),
       "valsel_mean": float(vs.mean()), "valsel_sd": float(vs.std(ddof=1)) if len(vs) > 1 else 0.0,
       "valsel_max": float(vs.max()),
       "cancel_auc_by_it": cr_auc, "vt_auc_by_it": vt_auc,
       "cancel_endpoint_auc": auc(cr_end),
       "cancel_at_800_per_run": {k: dict((d["it"], d) for d in runs[k]["info"]["diag"])
                                 .get(800, {}).get("cancel") for k in keys},
       "vt_at_800_per_run": {k: dict((d["it"], d) for d in runs[k]["info"]["diag"])
                             .get(800, {}).get("vt") for k in keys},
       "valsel_per_run": {k: hm(v) for k, v in runs.items()},
       "cancel_endpoint_per_run": {k: c for k, c in zip(keys, cr_end)},
       "bimodal": None}

verdict = {}
if 800 in cr_auc and cr_auc[800] is not None:
    verdict["SC_S1_pass"] = bool(cr_auc[800] >= 0.8)
    verdict["SC_S1_auc_cancel800"] = cr_auc[800]
    # SC-S2：中位数上半组好盆率
    cr8 = np.array([out["cancel_at_800_per_run"][k] for k in keys])
    med = np.median(cr8)
    upper = cr8 >= med
    verdict["SC_S2_pass"] = bool(good[upper].mean() >= 0.6)
    verdict["SC_S2_upper_good_rate"] = float(good[upper].mean())
if 800 in vt_auc and vt_auc[800] is not None:
    verdict["SC_S3_pass"] = bool(vt_auc[800] <= 0.6)
    verdict["SC_S3_auc_vt800"] = vt_auc[800]
verdict["SC_A_v4_mean"] = float(vs.mean())
verdict["SC_A_v4_pass"] = bool(vs.mean() >= 97.0)
if good.sum() and (~good).sum():
    gv, bv = vs[good], vs[~good]
    out["bimodal"] = {"good_center": float(gv.mean()), "bad_center": float(bv.mean()),
                      "gap": float(gv.min() - bv.max()),
                      "n_good": int(good.sum()), "n_bad": int((~good).sum())}
    verdict["bimodal"] = out["bimodal"]
out["verdict"] = verdict

tab = ["# T17 — B29d 早期盆筛选协议验证（PREREG_B29d）", "",
       "16 新 init × 流 4000，与 B29c cx_v2 同协议；每 100 步记录 cancel_ratio（无标签"
       "前向诊断）与 vt（train-class val）。", "",
       "| it | cancel AUC | vt AUC |", "|---|---|---|"]
for it in its:
    if it in cr_auc and cr_auc[it] is not None:
        tab.append(f"| {it} | {cr_auc[it]:.2f} | {vt_auc.get(it, float('nan')):.2f} |")
tab += ["", f"cancel_ratio@endpoint AUC: {auc(cr_end):.2f}（B29c 复现: 1.00）",
        f"n={len(vs)}，好盆(n≥93)={int(good.sum())}，valsel 均值 {vs.mean():.2f} ± "
        f"{vs.std(ddof=1) if len(vs) > 1 else 0:.2f}（max {vs.max():.1f}）",
        f"SC-S1(cancel@800 AUC≥0.8): {'PASS' if verdict.get('SC_S1_pass') else 'FAIL'}"
        f"（AUC={verdict.get('SC_S1_auc_cancel800', float('nan')):.2f}）",
        f"SC-S2(上半组好盆率≥60%): {'PASS' if verdict.get('SC_S2_pass') else 'FAIL'}"
        f"（rate={verdict.get('SC_S2_upper_good_rate', 0):.0%}, 基线 33%）",
        f"SC-S3(vt@800 AUC≤0.6): {'PASS' if verdict.get('SC_S3_pass') else 'FAIL'}"
        f"（AUC={verdict.get('SC_S3_auc_vt800', float('nan')):.2f}）"]
if out["bimodal"]:
    b = out["bimodal"]
    tab += [f"双盆复现: 好盆 {b['good_center']:.1f}(n={b['n_good']}) vs "
            f"坏盆 {b['bad_center']:.1f}(n={b['n_bad']})"]
tab.append("")
tab.append("逐 run: | run | cancel@800 | vt@800 | cancel@end | valsel |")
tab.append("|---|---|---|---|---|")
for k in keys:
    tab.append(f"| {k} | {out['cancel_at_800_per_run'][k]:.3f} | "
               f"{out['vt_at_800_per_run'][k]:.3f} | {out['cancel_endpoint_per_run'][k]:.3f} "
               f"| {out['valsel_per_run'][k]:.1f} |")
os.makedirs(os.path.join(BASE, "04_results", "tables"), exist_ok=True)
with open(os.path.join(BASE, "04_results", "tables", "T17_b29d_screening.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(tab))
json.dump(out, open(os.path.join(BASE, "04_results", "logs", "B29d_verdict.json"), "w"),
          indent=1)
print(json.dumps(verdict, indent=1, ensure_ascii=False))
print("\n".join(tab[7:14]))
