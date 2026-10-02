"""B29 分析：SC-A-v2 判定（PREREG_B29）+ 管道修复收益量化 → T15/B29_verdict.json。

判定（服务器运行前已预注册）：
  SC-A-v2  cx_v2 8 runs 的 valsel 权重 orbital 头 3σ 均值的均值 ≥ 97.0
  SC-V-v2  同指标 runs 间 sd ≤ 5.0
  锚       real_v2 3 种子同协议均值；gap 及逐 σ CI
  SC-AUG   cxpn_v2 − cx_v2（同 init 配对）均值差 > +1.0 → 增强有效；|差|≤1 → 无增益
  F2 复核  val_curve 里 best_it 分布（>1200 的比例 = 终点快照伪象证据）；
           valsel − endpoint 的测试增益
  EMA 复核 ema vs endpoint vs valsel 三权重集对比
  噪声探针 同种子重跑差
"""
import json, os, sys
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.join(BASE, "04_results", "logs", "B29_sca_v2.json")
res = json.load(open(path))
runs = res["runs"]
SIG = sorted({float(k.split("_")[1]) for k in
              next(iter(runs.values()))["eval"]["valsel"] if k.startswith("sigma_")})


def head_mean(entry, wset, head):
    vals = [entry["eval"][wset][f"sigma_{s:.2f}"][head] for s in SIG]
    return 100.0 * float(np.mean(vals))


def head_by_sigma(entry, wset, head):
    return {s: 100.0 * entry["eval"][wset][f"sigma_{s:.2f}"][head] for s in SIG}


cx = {k: v for k, v in runs.items() if k.startswith("cx_v2_")}
cxpn = {k: v for k, v in runs.items() if k.startswith("cxpn_v2_")}
real = {k: v for k, v in runs.items() if k.startswith("real_v2_")}

out = {"meta": res["meta"], "n_cx": len(cx), "n_cxpn": len(cxpn), "n_real": len(real)}

# ---- cx_v2：三权重集 ----
for wset in ("valsel", "endpoint", "ema"):
    ms = [head_mean(v, wset, "orbital") for v in cx.values()
          if wset in v["eval"]]
    if ms:
        out[f"cx_{wset}_mean"] = float(np.mean(ms))
        out[f"cx_{wset}_sd"] = float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0
        out[f"cx_{wset}_per_run"] = {k: head_mean(v, wset, "orbital")
                                     for k, v in cx.items() if wset in v["eval"]}
        out[f"cx_{wset}_flat"] = {k: float(max(head_by_sigma(v, wset, "orbital").values())
                                           - min(head_by_sigma(v, wset, "orbital").values()))
                                  for k, v in cx.items() if wset in v["eval"]}

# ---- SC 判定 ----
verdict = {}
if "cx_valsel_mean" in out:
    verdict["SC_A_v2_pass"] = bool(out["cx_valsel_mean"] >= 97.0)
    verdict["SC_A_v2_mean"] = out["cx_valsel_mean"]
    verdict["SC_V_v2_pass"] = bool(out["cx_valsel_sd"] <= 5.0)
    verdict["SC_V_v2_sd"] = out["cx_valsel_sd"]

# ---- real 锚 ----
rm = [head_mean(v, "valsel", "euclid") for v in real.values() if "valsel" in v["eval"]]
re_ep = [head_mean(v, "endpoint", "euclid") for v in real.values()]
out["real_valsel_mean"] = float(np.mean(rm)) if rm else None
out["real_endpoint_mean"] = float(np.mean(re_ep)) if re_ep else None
if rm and "cx_valsel_mean" in out:
    out["gap_real_minus_cx"] = float(np.mean(rm) - out["cx_valsel_mean"])
    verdict["gap_le_2pt"] = bool(out["gap_real_minus_cx"] <= 2.0)

# ---- SC-AUG（同 init 配对 cxpn − cx，流 f0）----
pairs = {}
for k in cxpn:
    base = k.replace("cxpn_v2_", "")
    ck = f"cx_v2_{base}"
    if ck in cx:
        pairs[base] = head_mean(cxpn[k], "valsel", "orbital") - head_mean(
            cx[ck], "valsel", "orbital")
if pairs:
    diffs = np.array(list(pairs.values()))
    out["aug_paired_diff"] = {k: float(v) for k, v in pairs.items()}
    out["aug_mean_diff"] = float(np.mean(diffs))
    if len(diffs) >= 2:
        t = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        from scipy import stats
        out["aug_t"] = float(t)
        # ttest_1samp 默认 alternative='two-sided'，其 pvalue 已是双侧 p 值，
        # 不可再乘 2（2026-10-03 审计轮修复：旧代码产出非法 aug_p=1.14>1）。
        out["aug_p"] = float(stats.ttest_1samp(diffs, 0).pvalue)
    verdict["SC_AUG"] = ("enhancement" if out["aug_mean_diff"] > 1.0
                         else "no_gain" if abs(out["aug_mean_diff"]) <= 1.0
                         else "harmful")

# ---- F2 复核：best_it 分布 + valsel−endpoint 增益 ----
bits = [v["info"]["best_it"] for v in cx.values()]
out["best_it"] = bits
out["best_it_gt1200_frac"] = float(np.mean([b > 1200 for b in bits]))
out["valsel_minus_endpoint"] = float(np.mean(
    [head_mean(v, "valsel", "orbital") - head_mean(v, "endpoint", "orbital")
     for v in cx.values()]))

# ---- EMA 复核 ----
if "cx_ema_mean" in out:
    out["ema_vs_endpoint"] = out["cx_ema_mean"] - out.get("cx_endpoint_mean", float("nan"))

# ---- 逐 σ 表（T15）----
tab = ["# T15 — B29 管道修复版 SC-A 攻坚（PREREG_B29）", "",
       "协议：disjoint 采样 + 对齐 val 选点（sp['val'] 未见类 2-way，σ 注入均值）+ "
       "全程 checkpoint + 修复 EMA + init×流解耦 + 确定性开关。"
       "绝对精度与 B25 不可直接比较（disjoint 修复）。", "",
       "| run | init | stream | valsel | endpoint | ema | best_it | flat(valsel) |",
       "|---|---|---|---|---|---|---|---|"]
for k, v in sorted(cx.items()):
    row = [k, k.split("_i")[1].split("_")[0], k.split("_f")[1],
           f"{head_mean(v, 'valsel', 'orbital'):.1f}",
           f"{head_mean(v, 'endpoint', 'orbital'):.1f}",
           f"{head_mean(v, 'ema', 'orbital'):.1f}" if "ema" in v["eval"] else "—",
           str(v["info"]["best_it"]),
           f"{out['cx_valsel_flat'][k]:.2f}"]
    tab.append("| " + " | ".join(row) + " |")
tab += ["", f"cx_v2 均值(valsel): {out.get('cx_valsel_mean', 0):.2f} ± "
        f"{out.get('cx_valsel_sd', 0):.2f} (sd)"]
if out.get("real_valsel_mean"):
    tab += [f"real_v2 均值(valsel, euclid头): {out['real_valsel_mean']:.2f} "
            f"(endpoint {out.get('real_endpoint_mean', 0):.2f})",
            f"gap (real − cx): {out.get('gap_real_minus_cx', 0):.2f} pt"]
if "aug_mean_diff" in out:
    tab += [f"cxpn − cx 配对均值差: {out['aug_mean_diff']:.2f} pt "
            f"(p={out.get('aug_p', float('nan')):.3f}) → SC_AUG={verdict.get('SC_AUG')}"]
if "ema_vs_endpoint" in out:
    tab += [f"ema − endpoint: {out['ema_vs_endpoint']:.2f} pt"]
tab += ["", f"best_it>1200 比例: {out['best_it_gt1200_frac']:.2f} "
        f"(F2: 终点快照伪象证据)", ""]
os.makedirs(os.path.join(BASE, "04_results", "tables"), exist_ok=True)
with open(os.path.join(BASE, "04_results", "tables", "T15_b29_sca_v2.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(tab))

# ---- 噪声探针 ----
if "noise_probe" in res and res["noise_probe"]:
    # 噪声探针的正确解读（B29 noise mode 的设计：rerun 块对每个匹配 key 都以
    # 固定 seed 11 重训同一 trunk → 三条记录逐位一致 = 确定性协议下的完全复现，
    # delta 恒 0）。run 间差异全部来自种子/流（盆），与 cuDNN 非确定性无关。
    np_out = {"note": "deterministic mode: same-seed rerun reproduces bit-exact; "
                      "inter-run variance is seed/stream (basin) level only",
              "n_rerun_records": len(res["noise_probe"]),
              "all_identical": len({json.dumps(v, sort_keys=True)
                                    for v in res["noise_probe"].values()}) == 1}
    out["noise_probe"] = np_out

verdict["all_prereg_criteria"] = {
    k: v for k, v in verdict.items() if k.startswith(("SC_", "gap"))}
out["verdict"] = verdict
json.dump(out, open(os.path.join(BASE, "04_results", "logs", "B29_verdict.json"), "w"),
          indent=1)
print(json.dumps(verdict, indent=1, ensure_ascii=False))
print("\n".join(tab[-12:]))
