#!/usr/bin/env python3
"""B-J3: build the real-control (98.60) row for paper B Table 1.

Source = 04_results/logs/B29_sca_v2.json real_v2 runs (9 paired runs,
init {11,23,37} x stream {0,12,26}), identical repaired protocol to the
equivariant cx_v2 arm: 3200 steps, disjoint episodes, deterministic mode,
val_train selection, sigma-paired 500-episode evaluation per point.
Head = euclidean (the non-equivariant control's natural readout).
Emits T27_real_control.md + real_control_row.json.
"""
import json
import statistics as st
from pathlib import Path

LOGS = Path(r"D:/个人/CVCNN/CVFEW/04_results/logs")
SRC = LOGS / "B29_sca_v2.json"
d = json.loads(SRC.read_text())
runs = d["runs"]
meta = d["meta"]

ks = sorted(k for k in runs if k.startswith("real_v2"))
HEAD = "euclid"
SELS = ["valsel", "endpoint", "ema"]

per_run = {}
for k in ks:
    ev = runs[k]["eval"]
    per_run[k] = {}
    for sel in SELS:
        sigmas = [s for s in ev[sel] if s.startswith("sigma_")]
        if not sigmas:
            continue
        per_run[k][sel] = {
            "n_sigma": len(sigmas),
            "mean_acc": st.mean(ev[sel][s][HEAD] for s in sigmas),
            "by_sigma": {s: ev[sel][s][HEAD] for s in sigmas},
        }

summary = {}
for sel in SELS:
    vals = [per_run[k][sel]["mean_acc"] for k in ks if sel in per_run[k]]
    if not vals:
        continue
    summary[sel] = {
        "n": len(vals),
        "mean_pct": round(100 * st.mean(vals), 2),
        "sd_pct": round(100 * (st.stdev(vals) if len(vals) > 1 else 0.0), 2),
        "min_pct": round(100 * min(vals), 2),
        "max_pct": round(100 * max(vals), 2),
        "per_run_pct": [round(100 * v, 2) for v in vals],
    }

payload = {
    "source": str(SRC),
    "head": HEAD,
    "protocol": {
        "arm": "real_v2: RealAMC real-valued non-equivariant control (2x params, euclid head)",
        "n_runs": len(ks),
        "inits": meta["init_seeds"],
        "streams": sorted({int(k.split("_f")[-1]) for k in ks}),
        "steps": 3200,
        "budget_episodes_per_step": meta["budget"],
        "selection": "val_train train-class 5-way signal (calibrated B29b) for valsel; endpoint/ema as named",
        "eval": "500 episodes per sigma; sigma-paired seeds; disjoint support/query",
        "deterministic": bool(meta["deterministic"]),
        "split": "split-0, same classes/protocol as B32 invpow_metric composite arm",
        "params": runs[ks[0]]["params"],
    },
    "summary": summary,
    "per_run": per_run,
}
(LOGS / "real_control_row.json").write_text(json.dumps(payload, indent=1))

lines = [
    "# T27 real-control row for Table 1 (concern B-J3 closure)\n",
    "Source file: `04_results/logs/B29_sca_v2.json`, arm `real_v2`.\n",
    "## Protocol (now disclosed in paper B Table 1 note)",
    f"- Architecture: RealAMC real-valued non-equivariant trunk, 2x parameters of the",
    f"  complex equivariant trunk ({runs[ks[0]]['params']} params), euclidean head;",
    "  no equivariance, no invariant readout.",
    f"- Runs: n={len(ks)} paired runs = init seeds {{11, 23, 37}} x episode streams {{0, 12, 26}}.",
    f"- Training: 3200 steps, {meta['budget']}-episode budget per step, deterministic mode,",
    "  disjoint support/query frames (F4 repair).",
    "- Checkpoint selection: valsel = train-class 5-way validation signal",
    "  (calibrated in the B29b selection-signal study); endpoint and EMA",
    f"  (decay {meta['ema_decay']:.3f}, bias-corrected, parameters only) reported alongside.",
    "- Evaluation: 500 episodes per sigma point; sigma-paired seeds; three injected",
    "  phase widths {0, 1.05, pi}; mean over the sigma axis reported.",
    "- Split: split-0, identical unseen classes and protocol to the composite arm",
    "  (B32 `invpow_metric`), so the 98.86 vs 98.60 comparison is protocol-matched.\n",
    "## Numbers (percent)",
    "| selection | n | mean | sd | min | max |",
    "|---|---|---|---|---|---|",
]
for sel in SELS:
    if sel in summary:
        s = summary[sel]
        lines.append(f"| {sel} | {s['n']} | {s['mean_pct']:.2f} | {s['sd_pct']:.2f} | "
                     f"{s['min_pct']:.2f} | {s['max_pct']:.2f} |")
lines.append("\nPer-run valsel (mean over sigma axis, %): " +
             " ".join(str(x) for x in summary["valsel"]["per_run_pct"]))
lines.append("\nPaper headline 98.60 = valsel mean over sigma axis. The composite arm's")
lines.append("+0.26 pt margin (98.86 vs 98.60) is smaller than the real control's own")
lines.append(f"run-to-run spread (sd {summary['valsel']['sd_pct']:.2f} pt over 9 runs "
             "across 3 inits x 3 streams),")
lines.append("so the paper states 'on par' rather than 'better', which is the honest")
lines.append("reading of the paired evidence.")
Path(r"D:/个人/CVCNN/CVFEW/04_results/tables/T27_real_control.md").write_text("\n".join(lines))
print(json.dumps(summary, indent=1))
print("params:", runs[ks[0]]["params"])
print("wrote T27_real_control.md + real_control_row.json")
