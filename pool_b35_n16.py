"""B35 n=8→16 池化分析：合并 B35（8 init）+ B35b（8 新 init，offset=8）的
real/midres 臂，做 16 配对的 Wilcoxon 检验，目标把 midres−real 的 p=0.055
缩到 <0.05。

输入：
  04_results/logs/B35_real.json / B35_midres.json      （原 n=8）
  04_results/logs/B35b_real.json / B35b_midres.json     （新 n=8，下载自服务器
      /tmp/cvfe_work/res_b35ext/logs/B35_{arm}.json）

输出：04_results/logs/B35_n16_pooled.json + 04_results/tables/T26_b35_n16.md
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(BASE, "04_results", "logs")


def load(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    runs = d.get("runs", d)
    out = {k: v for k, v in runs.items() if isinstance(v, dict)}
    return out


SIG = (0.0, 1.0471975511965976, 3.141592653589793)  # 与 analyze_b35.py / run_b35.py 同口径


def valsel(runs):
    """valsel 均值（百分比；与 B35 verdict / analyze_b35.py 同口径）。

    数据结构：runs[k]["eval"]["valsel"][f"sigma_{s:.2f}"]["euclid"]，s ∈ SIG。
    键按 init 种子编号归一（k.split("_i")[-1]），保证 B35 与 B35b 批次合并后
    同一 init 的 midres 与 real 可配对。
    """
    out = {}
    for k, v in runs.items():
        ev = v.get("eval", {})
        sel = ev.get("valsel")
        if not isinstance(sel, dict):
            continue
        try:
            a = [100 * sel[f"sigma_{s:.2f}"]["euclid"] for s in SIG]
        except (KeyError, TypeError):
            continue
        out[str(k.split("_i")[-1])] = sum(a) / len(a)
    return out


def paired(a, b):
    """a, b 已为百分比。按 init 配对做 Wilcoxon。"""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    p = float(wilcoxon(d).pvalue) if len(d) > 3 and not np.allclose(d, 0) else None
    return {"n": len(keys), "mean_d": float(np.mean(d)),
            "win_rate": float(np.mean(d > 0)), "p_wilcoxon": p,
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    arms = {}
    for tag, fn in (("b35", "B35_{arm}.json"), ("b35b", "B35b_{arm}.json")):
        for arm in ("real", "midres"):
            r = load(fn.format(arm=arm))
            if r:
                acc = valsel(r)
                if acc:
                    arms[(tag, arm)] = acc
                    print(f"{tag}/{arm}: {len(acc)} runs")
    real = {**arms.get(("b35", "real"), {}), **arms.get(("b35b", "real"), {})}
    mid = {**arms.get(("b35", "midres"), {}), **arms.get(("b35b", "midres"), {})}
    # 按 init 对齐：real_i47 ↔ midres_i47
    pairs_real, pairs_mid = {}, {}
    for k in real:
        i = k.split("_i")[1] if "_i" in k else k
        pairs_real[i] = real[k]
    for k in mid:
        i = k.split("_i")[1] if "_i" in k else k
        pairs_mid[i] = mid[k]
    common = sorted(set(pairs_real) & set(pairs_mid))
    print(f"paired inits: {len(common)}")
    par = paired(pairs_real, pairs_mid)
    out = {"n_pooled": len(common), "inits": common,
           "midres_minus_real_pooled": par,
           "mean_real": float(np.mean([pairs_real[i] for i in common])),
           "mean_midres": float(np.mean([pairs_mid[i] for i in common])),
           "verdict_sc_B35_midres_n16": bool(par["mean_d"] >= 2.0 and par["p_wilcoxon"] is not None
                                              and par["p_wilcoxon"] < 0.05)}
    json.dump(out, open(os.path.join(LOGS, "B35_n16_pooled.json"), "w"), indent=1)
    lines = ["# T26：B35 n=8→16 池化（midres−real 显著性补强）", "",
             f"配对 init 数：{len(common)}（B35 第1–8 + 新 9–16）", "",
             "| 口径 | 值 |", "|---|---|",
             f"| real 均值 | {out['mean_real']:.2f} |",
             f"| midres 均值 | {out['mean_midres']:.2f} |",
             f"| midres−real 配对差 | {par['mean_d']:+.2f} pt |",
             f"| 胜率 | {par['win_rate']*100:.0f}% ({int(par['win_rate']*len(common))}/{len(common)}) |",
             f"| Wilcoxon p | {par['p_wilcoxon']:.4f} |",
             f"| 95% CI | [{par['ci'][0]:+.2f}, {par['ci'][1]:+.2f}] |",
             f"| SC-B35-Midres(n=16) | {'PASS' if out['verdict_sc_B35_midres_n16'] else 'FAIL'} |", "",
             f"原 n=8：+2.43pt（7/8 胜，p=0.055）。n=16 目标：p<0.05。"]
    tb = os.path.join(BASE, "04_results", "tables", "T26_b35_n16.md")
    open(tb, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(out["midres_minus_real_pooled"], indent=1))
    print("->", tb)


if __name__ == "__main__":
    main()
