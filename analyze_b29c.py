"""B29c 分析：SC-A-v3 判定（PREREG_B29c）+ 盆分布报告 → T16/B29c_verdict.json。

判定（预注册）：
  SC-A-v3  cx_v2_vt 12 runs 的 valsel（val_train 校准选点）orbital 头 3σ 均值的均值 ≥ 97.0
  SC-V-v3  同指标 sd ≤ 6.0
  分布报告 好盆（valsel ≥ 95）比例 + 双峰性（1D 2-GMM 的重叠/间隔）
  选点收益  valsel − endpoint 分布
"""
import json, os
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
res = json.load(open(os.path.join(BASE, "04_results", "logs", "B29c_sca_v3.json")))
runs = res["runs"]
SIG = (0.0, np.pi / 3, np.pi)


def hm(entry, wset, head="orbital"):
    if wset not in entry["eval"]:
        return None
    return 100.0 * float(np.mean([entry["eval"][wset][f"sigma_{s:.2f}"][head]
                                  for s in SIG]))


cx = {k: v for k, v in runs.items() if k.startswith("cx_v2_vt")}
out = {"meta": res["meta"], "n_runs": len(cx)}
for wset in ("valsel", "endpoint", "ema"):
    ms = [hm(v, wset) for v in cx.values()]
    ms = [m for m in ms if m is not None]
    if ms:
        out[f"cx_{wset}"] = {"mean": float(np.mean(ms)),
                             "sd": float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
                             "max": float(np.max(ms)), "min": float(np.min(ms)),
                             "per_run": {k: hm(v, wset) for k, v in cx.items()
                                         if hm(v, wset) is not None}}

verdict = {}
if "cx_valsel" in out:
    v = out["cx_valsel"]
    verdict["SC_A_v3_pass"] = bool(v["mean"] >= 97.0)
    verdict["SC_A_v3_mean"] = v["mean"]
    verdict["SC_V_v3_pass"] = bool(v["sd"] <= 6.0)
    verdict["SC_V_v3_sd"] = v["sd"]
    # 好盆比例 + 双峰性
    vals = np.array(sorted(v["per_run"].values()))
    verdict["good_basin_ge95_frac"] = float(np.mean(vals >= 95))
    verdict["good_basin_ge93_frac"] = float(np.mean(vals >= 93))
    # 1D 2-GMM 双峰性：粗分（>=92 为好盆）报告两群中心与 gap
    good = vals[vals >= 92]
    bad = vals[vals < 92]
    if len(good) and len(bad):
        verdict["bimodal"] = {"good_center": float(good.mean()),
                              "bad_center": float(bad.mean()),
                              "gap": float(good.min() - bad.max()),
                              "n_good": int(len(good)), "n_bad": int(len(bad))}
    # 选点收益
    vs = np.array([hm(v_, "valsel") for v_ in cx.values()])
    ep = np.array([hm(v_, "endpoint") for v_ in cx.values()])
    verdict["sel_gain_valsel_minus_endpoint"] = {"mean": float(np.mean(vs - ep)),
                                                 "per_run": {
        k: float(hm(v_, "valsel") - hm(v_, "endpoint")) for k, v_ in cx.items()}}
    em = np.array([hm(v_, "ema") for v_ in cx.values()])
    verdict["ema_gain"] = float(np.mean(em - ep))
    verdict["flat_valsel_max"] = float(max(
        max(v_["eval"]["valsel"][f"sigma_{s:.2f}"]["orbital"] for s in (0.0, np.pi / 3, np.pi))
        - min(v_["eval"]["valsel"][f"sigma_{s:.2f}"]["orbital"] for s in (0.0, np.pi / 3, np.pi))
        for v_ in cx.values()))

# F2 复核（val_train 版）：best_it 分布
bits = [v["info"]["best_it"] for v in cx.values()]
verdict["best_it"] = bits
verdict["best_it_gt1200_frac"] = float(np.mean([b > 1200 for b in bits]))

out["verdict"] = verdict

# ---- T16 表 ----
tab = ["# T16 — B29c 终版 SC-A（校准选点信号 val_train，PREREG_B29c）", "",
       "协议：与 B29 cx_v2 同（big ch/l1pca5/3200 步/disjoint/确定性/500 epi 测试），"
       "选点信号 = 训练类 5-way 5-shot val（B29b 校准：Spearman 0.79-0.83，不饱和）。", "",
       "| run | init | stream | valsel | endpoint | ema | best_it |", "|---|---|---|---|---|---|---|"]
for k, v in sorted(cx.items()):
    row = [k, k.split("_i")[1].split("_")[0], k.split("_f")[1],
           f"{hm(v, 'valsel'):.1f}" if hm(v, "valsel") is not None else "—",
           f"{hm(v, 'endpoint'):.1f}", f"{hm(v, 'ema'):.1f}" if hm(v, "ema") is not None else "—",
           str(v["info"]["best_it"])]
    tab.append("| " + " | ".join(row) + " |")
if "cx_valsel" in out:
    v = out["cx_valsel"]
    tab += ["", f"valsel 均值 {v['mean']:.2f} ± {v['sd']:.2f}（max {v['max']:.1f} / "
            f"min {v['min']:.1f}）",
            f"好盆(≥95)比例: {verdict['good_basin_ge95_frac']:.2f}；"
            f"SC_A_v3={'PASS' if verdict['SC_A_v3_pass'] else 'FAIL'}（阈值 97.0）",
            f"选点收益 valsel−endpoint: {verdict['sel_gain_valsel_minus_endpoint']['mean']:+.2f} pt",
            f"EMA−endpoint: {verdict.get('ema_gain', 0):+.2f} pt"]
    if "bimodal" in verdict:
        b = verdict["bimodal"]
        tab += [f"双峰: 好盆中心 {b['good_center']:.1f}(n={b['n_good']}) vs "
                f"坏盆中心 {b['bad_center']:.1f}(n={b['n_bad']})，gap {b['gap']:.1f}pt"]
os.makedirs(os.path.join(BASE, "04_results", "tables"), exist_ok=True)
with open(os.path.join(BASE, "04_results", "tables", "T16_b29c_sca_v3.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(tab))
json.dump(out, open(os.path.join(BASE, "04_results", "logs", "B29c_verdict.json"), "w"),
          indent=1)
print(json.dumps(verdict, indent=1, ensure_ascii=False))
print("\n".join(tab[-10:]))
