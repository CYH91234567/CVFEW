"""独立复算论文 B 头条数字（防多轮修订漂移）。

逐条从 04_results/logs 原 JSON 重算，与 latex_B main.tex 的陈述比对：
  TB-1  Table~ref{tab:five} 五臂面板（split-0，n=16）
  TB-2  comp − gated = +16.01pt，16/16，p=3.05e-5（n=16）
  TB-3  real control 98.60±0.35（n=9）
  TB-4  cross-split：comp−gated +33.67pt，8/8，p=0.008；comp 81.12±3.44；gated 47.45
  TB-5  B35：midres 84.76±2.62、real 82.33±1.40、+2.43pt、7/8、p=0.055、δinv≤4.3e-5
        （论文佐证数字；n=16 池化后由 pool_b35_n16.py 另行产出）
输出：每条 PASS/FAIL + 逐位差。FAIL 即存在数字漂移，须回查。
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(BASE, "04_results", "logs")
SIG = (0.0, 1.0471975511965976, 3.141592653589793)


def runs(name):
    return json.load(open(os.path.join(LOGS, name)))["runs"]


def accs(rs, sel="valsel", head="euclid"):
    out = {}
    for k, v in rs.items():
        ev = v["eval"]
        if sel in ev:
            try:
                a = [100 * ev[sel][f"sigma_{s:.2f}"][head] for s in SIG]
            except KeyError:
                continue
            out[k.split("_i")[-1]] = float(np.mean(a))
    return out


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    p = float(wilcoxon(d).pvalue) if len(d) > 3 and not np.allclose(d, 0) else None
    return {"n": len(keys), "mean_d": float(np.mean(d)),
            "wins": int(np.sum(np.array(d) > 0)), "p": p}


def check(tag, computed, stated, tol=0.06):
    ok = abs(computed - stated) <= tol
    shown = f"{computed:.2e}" if computed is not None and abs(computed) < 1e-3 else f"{computed:.4f}"
    print(f"{'PASS' if ok else 'FAIL'}  {tag}: computed {shown} vs stated {stated} (tol {tol})")
    return ok


def check_str(tag, computed_str, stated_str):
    ok = computed_str == stated_str
    print(f"{'PASS' if ok else 'FAIL'}  {tag}: computed {computed_str} vs stated {stated_str}")
    return ok


def main():
    allok = True
    # TB-1/2：split-0 五臂
    b32 = {a: runs(f"B32_{a}.json") for a in ("metric", "invpow", "invpow_metric")}
    gated0 = runs("B30_gated.json")
    gated0 = {k: v for k, v in gated0.items() if k.startswith("gated_i")}
    for arm, stated_m, stated_s in (("metric", 98.76, 0.36), ("invpow", 86.19, 0.96),
                                    ("invpow_metric", 98.86, 0.29),
                                    ("autocorr", 85.80, 5.42)):
        rs = b32[arm] if arm in b32 else runs("B30_autocorr.json")
        a = accs(rs)
        allok &= check(f"TB-1 {arm} mean", float(np.mean(list(a.values()))), stated_m)
        allok &= check(f"TB-1 {arm} sd", float(np.std(list(a.values()), ddof=1)), stated_s, tol=0.08)
    g0 = accs(gated0, head="orbital")   # gated 的自然头是 orbital（B33 报告口径）
    allok &= check("TB-1 gated mean", float(np.mean(list(g0.values()))), 82.85, tol=0.15)
    allok &= check("TB-1 gated sd", float(np.std(list(g0.values()), ddof=1)), 10.30, tol=0.15)

    cmp0 = accs(b32["invpow_metric"])
    gg0 = accs(gated0, head="orbital")
    pr = paired(cmp0, gg0)
    allok &= check("TB-2 comp−gated mean_d", pr["mean_d"], 16.01, tol=0.1)
    allok &= check_str("TB-2 wins", f"{pr['wins']}/{pr['n']}", "16/16")
    allok &= check("TB-2 p", pr["p"], 3.05e-5, tol=2e-5)

    # TB-3：real control
    rc = json.load(open(os.path.join(LOGS, "real_control_row.json")))["summary"]["valsel"]
    allok &= check("TB-3 real mean", rc["mean_pct"], 98.60, tol=0.02)
    allok &= check("TB-3 real sd", rc["sd_pct"], 0.35, tol=0.03)

    # TB-4：cross-split
    b33c, b33g = runs("B33_invpow_metric.json"), runs("B33_gated.json")
    c33, g33 = accs(b33c), accs(b33g, head="orbital")
    pr33 = paired(c33, g33)
    allok &= check("TB-4 comp−gated split-B", pr33["mean_d"], 33.67, tol=0.15)
    allok &= check_str("TB-4 wins", f"{pr33['wins']}/{pr33['n']}", "8/8")
    allok &= check("TB-4 p", pr33["p"], 0.008, tol=0.002)
    allok &= check("TB-4 comp mean", float(np.mean(list(c33.values()))), 81.12, tol=0.15)
    allok &= check("TB-4 comp sd", float(np.std(list(c33.values()), ddof=1)), 3.44, tol=0.15)
    allok &= check("TB-4 gated mean", float(np.mean(list(g33.values()))), 47.45, tol=0.15)

    # TB-5：B35 n=8
    real8, mid8 = runs("B35_real.json"), runs("B35_midres.json")
    r8, m8 = accs(real8), accs(mid8)
    allok &= check("TB-5 midres mean", float(np.mean(list(m8.values()))), 84.76, tol=0.15)
    allok &= check("TB-5 midres sd", float(np.std(list(m8.values()), ddof=1)), 2.62, tol=0.15)
    allok &= check("TB-5 real mean", float(np.mean(list(r8.values()))), 82.33, tol=0.15)
    allok &= check("TB-5 real sd", float(np.std(list(r8.values()), ddof=1)), 1.40, tol=0.15)
    pr35 = paired(m8, r8)
    allok &= check("TB-5 midres−real", pr35["mean_d"], 2.43, tol=0.1)
    allok &= check_str("TB-5 wins", f"{pr35['wins']}/{pr35['n']}", "7/8")
    allok &= check("TB-5 p", pr35["p"], 0.055, tol=0.006)
    dinv = [v["info"]["invariance_delta"] for v in mid8.values()]
    dmax = max(dinv)
    allok &= check("TB-5 δinv max", dmax, 4.3e-5, tol=1e-5)

    print("\nALL PASS" if allok else "\n*** DRIFT DETECTED — see FAIL lines ***")


if __name__ == "__main__":
    main()
