"""B29d 分析：早期盆筛选判定（PREREG_B29d）→ T17/B29d_verdict.json。

 SC-S1  cancel_ratio@800 对好盆（valsel≥93）的 AUC ≥ 0.8（Mann-Whitney）
 SC-S2  cancel_ratio@800 中位数上半组的好盆率 ≥ 60%（基线=全体好盆率）
 SC-S3  vt@800 无**方向稳定**预测力：全部时点 AUC 与 0.5 的偏离 ≤ 2SE 且
        符号跨时点不稳定（2026-10-02 审计修正：旧判据"AUC≤0.6"把强反相关
        （AUC=0.07，|AUC−0.5|=0.43）误判为"无预测力"；反相关≠无信息，但
        方向翻转使其不可部署——判据改为显式的方向稳定性检验）
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


order = list(runs.keys())              # 插入序：vs/cr/upper 全部统一用此序
keys = sorted(runs)                    # 仅用于展示
vs = np.array([hm(runs[k]) for k in order])
ep = np.array([hm(runs[k], "endpoint") for k in order])
good = vs >= 93


def auc(scores):
    s = np.asarray(scores, float)
    if good.sum() == 0 or good.sum() == len(good):
        return None
    n1, n2 = good.sum(), (~good).sum()
    a = mannwhitneyu(s[good], s[~good]).statistic / (n1 * n2)
    se = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))     # H0 下的 AUC 标准误
    return a, se


its = [100, 200, 400, 600, 800, 1000, 1600, 2400, 3200]
cr_by_it = {it: [dict((d["it"], d) for d in runs[k]["info"]["diag"]).get(it, {}).get("cancel")
                 for k in order] for it in its}
vt_by_it = {it: [dict((d["it"], d) for d in runs[k]["info"]["diag"]).get(it, {}).get("vt")
                 for k in order] for it in its}
cr_auc = {it: auc(cr_by_it[it]) for it in its if all(x is not None for x in cr_by_it[it])}
vt_auc = {it: auc(vt_by_it[it]) for it in its if all(x is not None for x in vt_by_it[it])}
cr_end = [runs[k]["info"]["cancel_endpoint"] for k in order]

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
       "cancel_endpoint_per_run": {k: out_c for k, out_c in zip(keys, cr_end)},
       "bimodal": None}

verdict = {}
if 800 in cr_auc and cr_auc[800] is not None:
    a800, _ = cr_auc[800]
    verdict["SC_S1_pass"] = bool(a800 >= 0.8)
    verdict["SC_S1_auc_cancel800"] = a800
    # SC-S2：中位数上半组好盆率（cr8 与 good 同序）
    cr8 = np.array([out["cancel_at_800_per_run"][k] for k in order])
    med = np.median(cr8)
    upper = cr8 >= med
    verdict["SC_S2_pass"] = bool(good[upper].mean() >= 0.6)
    verdict["SC_S2_upper_good_rate"] = float(good[upper].mean())
    verdict["SC_S2_base_rate"] = float(good.mean())
if 800 in vt_auc and vt_auc[800] is not None:
    a800, se800 = vt_auc[800]
    # 方向稳定性：全部时点偏离 0.5 ≤ 2SE（不可与随机区分）；且跨时点符号不稳定
    dev = {it: (abs(vt_auc[it][0] - 0.5), vt_auc[it][1]) for it in vt_auc}
    no_signal = all(d <= 2 * s for d, s in dev.values())
    dirs = [vt_auc[it][0] - 0.5 for it in vt_auc]
    sign_unstable = (min(dirs) < 0 < max(dirs))
    verdict["SC_S3_pass"] = bool(no_signal and sign_unstable)
    verdict["SC_S3_auc_vt800"] = a800
    verdict["SC_S3_max_dev_over_se"] = float(max(d / s for d, s in dev.values()))
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
       "前向诊断）与 vt（train-class val）。AUC = Mann-Whitney；括号内为 H0 下 SE"
       f"（n_good={int(good.sum())}, n_bad={int((~good).sum())}）。", "",
       "| it | cancel AUC | vt AUC |", "|---|---|---|"]
for it in its:
    if it in cr_auc and cr_auc[it] is not None:
        ca, cs = cr_auc[it]
        va = vt_auc.get(it)
        vs_ = f"{va[0]:.2f} (±{va[1]:.2f})" if va else "—"
        tab.append(f"| {it} | {ca:.2f} (±{cs:.2f}) | {vs_} |")
ce = auc(cr_end)
tab += ["", f"cancel_ratio@endpoint AUC: {ce[0]:.2f} (±{ce[1]:.2f})（B29c 复现: 1.00）",
        f"n={len(vs)}，好盆(n≥93)={int(good.sum())}，valsel 均值 {vs.mean():.2f} ± "
        f"{vs.std(ddof=1) if len(vs) > 1 else 0:.2f}（max {vs.max():.1f}）",
        f"SC-S1(cancel@800 AUC≥0.8): {'PASS' if verdict.get('SC_S1_pass') else 'FAIL'}"
        f"（AUC={verdict.get('SC_S1_auc_cancel800', float('nan')):.2f}）",
        f"SC-S2(上半组好盆率≥60%): {'PASS' if verdict.get('SC_S2_pass') else 'FAIL'}"
        f"（rate={verdict.get('SC_S2_upper_good_rate', 0):.0%}, 基线={verdict.get('SC_S2_base_rate', 0):.0%}）",
        f"SC-S3(vt 无方向稳定预测力): {'PASS' if verdict.get('SC_S3_pass') else 'FAIL'}"
        f"（vt@800 AUC={verdict.get('SC_S3_auc_vt800', float('nan')):.2f}, "
        f"max|AUC−0.5|/SE={verdict.get('SC_S3_max_dev_over_se', float('nan')):.2f}; "
        f"旧判据 AUC≤0.6 会把 AUC=0.07 的强反相关误读为'无预测力'）"]
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
