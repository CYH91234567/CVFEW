"""B40 判定脚本：按 PREREG_B40.md 四判据分析 10b 四臂，并与 10a split-B
同协议结果做跨数据集对比。产物 B40_verdict.json + T30_b40_10b.md。

判据口径与 analyze_b35.py / pool_b35_n16.py 一致：valsel 三头均值（SIG 三 σ
平均），real/comp/midres=euclid 头、gated=orbital 头（与 B33 报告口径一致）。
"""
import json
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")
TAB = os.path.join(BASE, "04_results", "tables")
SIG = (0.0, 1.0471975511965976, 3.141592653589793)
ARM_HEAD = {"real": "euclid", "gated": "orbital",
            "comp": "euclid", "midres": "euclid"}


def load(name):
    p = os.path.join(LOG, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))["runs"]


def accs(runs, sel="valsel", head="euclid"):
    """valsel 下三 σ 平均（百分比）。键按 init 编号归一，便跨臂数据集配对。"""
    out = {}
    for k, v in runs.items():
        ev = v["eval"]
        if sel in ev:
            try:
                a = [100 * ev[sel][f"sigma_{s:.2f}"][head] for s in SIG]
            except KeyError:
                continue
            out[str(k.split("_i")[-1])] = sum(a) / len(a)
    return out


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    p = float(wilcoxon(d).pvalue) if len(d) > 3 and not np.allclose(d, 0) else None
    return {"n": len(keys), "keys": keys, "d": d,
            "mean_d": float(np.mean(d)) if d else None,
            "win_rate": float(np.mean([x > 0 for x in d])) if d else None,
            "p_wilcoxon": p,
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else None}


def stats(runs, sel="valsel", head="euclid"):
    a = list(accs(runs, sel, head).values())
    if not a:
        return {"n": 0}
    return {"n": len(a), "mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def certs(runs):
    """δ_inv（构造性不变证书）与 σ 平坦（决策级）。"""
    dinv = [v["info"]["invariance_delta"] for v in runs.values()
            if "invariance_delta" in v["info"]]
    flats = []
    for v in runs.values():
        ev = v["eval"]
        if "valsel" in ev:
            a = [100 * ev["valsel"][f"sigma_{s:.2f}"]["euclid"] for s in SIG]
            flats.append(max(a) - min(a))
    return {"delta_inv_max": float(np.max(dinv)) if dinv else None,
            "delta_inv_mean": float(np.mean(dinv)) if dinv else None,
            "sigma_flat_max_pt": float(np.max(flats)) if flats else None,
            "n": len(dinv)}


def main():
    arms = {}
    for arm in ("real", "gated", "comp", "midres"):
        arms[arm] = load(f"B40_{arm}.json")
    missing = [a for a, v in arms.items() if v is None]
    if missing:
        raise SystemExit(f"missing B40 arms: {missing}")

    out = OrderedDict()
    out["protocol"] = ("RML2016.10b fixed 5/2/3 split (test {8PSK,QAM64,AM-DSB}), "
                       "8 paired inits {7,13,17,19,29,31,41,43} x stream 4000, "
                       "3200 steps, val_train selection, sigma-paired eval, disjoint, "
                       "deterministic — identical to B32/B33/B35 (10a)")
    out["stats"] = {arm: stats(runs, head=ARM_HEAD[arm]) for arm, runs in arms.items()}
    # 跨数据集参照（10a split-B 同协议：B33 comp/gated + B35 real/midres）
    ref = {}
    for tag, fn, head in (("comp_10a", "B33_invpow_metric.json", "euclid"),
                          ("gated_10a", "B33_gated.json", "orbital"),
                          ("real_10a", "B35_real.json", "euclid"),
                          ("midres_10a", "B35_midres.json", "euclid")):
        r = load(fn)
        if r:
            ref[tag] = stats(r, head=head)

    # 判据 1：comp − gated（跨数据集复现）
    out["SC_B40_Replicate"] = paired(accs(arms["comp"], head="euclid"),
                                     accs(arms["gated"], head="orbital"))
    # 判据 2：parity（comp/midres − real）
    out["SC_B40_Parity_comp_minus_real"] = paired(accs(arms["comp"]),
                                                  accs(arms["real"]))
    out["SC_B40_Parity_midres_minus_real"] = paired(accs(arms["midres"]),
                                                    accs(arms["real"]))
    # 判据 3：midres − comp（方向性；≥+2pt & p<0.05 为强复现）
    out["SC_B40_Midres_midres_minus_comp"] = paired(accs(arms["midres"]),
                                                    accs(arms["comp"]))
    # 机制参照
    out["real_minus_gated"] = paired(accs(arms["real"]),
                                     accs(arms["gated"], head="orbital"))
    # 判据 4：构造性证书
    cert = {arm: certs(runs) for arm, runs in arms.items() if arm in ("comp", "midres")}
    out["SC_B40_Cert"] = cert

    # 10a 对照的平坦性（real：经验而非证书）
    rfl = []
    for v in arms["real"].values():
        ev = v["eval"]
        if "valsel" in ev:
            a = [100 * ev["valsel"][f"sigma_{s:.2f}"]["euclid"] for s in SIG]
            rfl.append(max(a) - min(a))
    out["real_sigma_flat_max_pt"] = float(np.max(rfl)) if rfl else None
    out["params"] = {arm: int(list(runs.values())[0]["info"]["params"])
                     for arm, runs in arms.items()}

    rep = out["SC_B40_Replicate"]
    par_c = out["SC_B40_Parity_comp_minus_real"]
    par_m = out["SC_B40_Parity_midres_minus_real"]
    mid = out["SC_B40_Midres_midres_minus_comp"]
    cert_ok = all(c["delta_inv_max"] is not None and c["delta_inv_max"] <= 1e-3
                  and c["sigma_flat_max_pt"] is not None
                  and c["sigma_flat_max_pt"] <= 0.1 for c in cert.values())
    verdict = {
        "SC_B40_Replicate": bool(rep["mean_d"] is not None and rep["mean_d"] >= 5.0
                                 and rep["p_wilcoxon"] is not None
                                 and rep["p_wilcoxon"] < 0.05
                                 and rep["win_rate"] >= 0.75),
        "SC_B40_Parity": bool(par_c["mean_d"] >= -2.0 and par_m["mean_d"] >= -2.0),
        "SC_B40_Midres_strong": bool(mid["mean_d"] >= 2.0 and mid["p_wilcoxon"] < 0.05),
        "SC_B40_Midres_directional": bool(mid["mean_d"] >= 0),
        "SC_B40_Cert": bool(cert_ok),
    }
    out["verdict"] = verdict
    out["cross_dataset_10a_splitB"] = ref
    out["per_run_valsel"] = {
        arm: {k: round(v, 3) for k, v in accs(runs, head=ARM_HEAD[arm]).items()}
        for arm, runs in arms.items()}

    with open(os.path.join(LOG, "B40_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)

    def f2(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"

    def fc(c, key, nd=2):
        return "—" if c is None or c[key] is None else f"{c[key]:.{nd}f}"

    lines = [
        "# T30 — B40：第二真实数据集（RML2016.10b）学习层四臂", "",
        "预注册：`02_plan/PREREG_B40.md`（判据运行前写定）。协议与 B32/B33/B35（10a）",
        "逐字一致，仅换数据集与类划分（固定 5/2/3，test={8PSK,QAM64,AM-DSB}，与 B34",
        "的 10b 确定性半边同 split）。8 配对 init×流 4000，3200 步，val_train 选点，",
        "σ 配对评估（SIG={0,π/3,π}），disjoint，确定性。", "",
        "## 四臂主面板", "",
        "| 臂 | n | valsel 均值 | sd | min | max | σ 平坦 max | δ_inv max | params |",
        "|---|---|---|---|---|---|---|---|---|"]
    for arm in ("real", "gated", "comp", "midres"):
        s, c = out["stats"][arm], cert.get(arm)
        flat = fc(c, "sigma_flat_max_pt", 3) if c else "—"
        dinv = ("%.2e" % c["delta_inv_max"]) if c and c["delta_inv_max"] is not None else "—"
        lines.append(
            f"| {arm} | {s['n']} | {f2(s['mean'])} | {f2(s['sd'])} | {f2(s['min'])} "
            f"| {f2(s['max'])} | {flat}pt | {dinv} | {out['params'][arm]} |")
    lines += ["", "## 配对比较（10b，n=8 配对 init）", "",
              "| 比较 | 均值差 | 胜率 | Wilcoxon p | 95% CI |",
              "|---|---|---|---|---|"]
    for name, key in (("SC-B40-Replicate：comp − gated", "SC_B40_Replicate"),
                      ("SC-B40-Parity：comp − real", "SC_B40_Parity_comp_minus_real"),
                      ("SC-B40-Parity：midres − real", "SC_B40_Parity_midres_minus_real"),
                      ("SC-B40-Midres：midres − comp", "SC_B40_Midres_midres_minus_comp"),
                      ("机制参照：real − gated", "real_minus_gated")):
        p = out[key]
        lines.append(f"| {name} | {f2(p['mean_d'])}pt | {p['n'] and int(p['win_rate']*p['n'])}/{p['n']} "
                     f"| {('%.4f' % p['p_wilcoxon']) if p['p_wilcoxon'] is not None else '—'} "
                     f"| [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] |")
    lines += ["", "## 跨数据集对照（10a split-B 同协议）", "",
              "| 臂 | 10a split-B | 10b | 差 |",
              "|---|---|---|---|"]
    pairs = (("comp", "comp_10a"), ("gated", "gated_10a"),
             ("real", "real_10a"), ("midres", "midres_10a"))
    for arm, tag in pairs:
        a, b = ref.get(tag, {}), out["stats"][arm]
        if a.get("n") and b.get("n"):
            lines.append(f"| {arm} | {f2(a['mean'])}±{f2(a['sd'])} (n={a['n']}) "
                         f"| {f2(b['mean'])}±{f2(b['sd'])} (n={b['n']}) "
                         f"| {b['mean']-a['mean']:+.2f}pt |")
    lines += ["", "## 判据小结（PREREG_B40）", "",
              f"- SC-B40-Replicate（comp−gated ≥ +5pt，p<0.05，≥6/8）："
              f"{'PASS' if verdict['SC_B40_Replicate'] else 'FAIL'} "
              f"({f2(rep['mean_d'])}pt，p={rep['p_wilcoxon'] and '%.4f' % rep['p_wilcoxon']})",
              f"- SC-B40-Parity（comp−real 与 midres−real 均 ≥ −2pt）："
              f"{'PASS' if verdict['SC_B40_Parity'] else 'FAIL'} "
              f"(comp {f2(par_c['mean_d'])}pt / midres {f2(par_m['mean_d'])}pt)",
              f"- SC-B40-Midres（方向性 ≥0；强 = ≥+2pt 且 p<0.05）："
              f"{'强复现' if verdict['SC_B40_Midres_strong'] else ('方向性 PASS' if verdict['SC_B40_Midres_directional'] else 'FAIL')} "
              f"({f2(mid['mean_d'])}pt)",
              f"- SC-B40-Cert（δ_inv ≤ 1e-3 且 σ 平坦 ≤ 0.1pt）："
              f"{'PASS' if verdict['SC_B40_Cert'] else 'FAIL'}",
              "",
      ("通过 ⇒ gap B-J1 闭合（第二真实数据集学习层证据，Evidence 4→5）；"
       "不过 ⇒ 按 PREREG §3 SC-B40-Negative-honest 分支诚实披露并在 limitation 收窄主张。"),
    ]
    open(os.path.join(TAB, "T30_b40_10b.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(verdict, indent=1))
    print("->", os.path.join(TAB, "T30_b40_10b.md"))


if __name__ == "__main__":
    main()
