#!/usr/bin/env python3
"""A-J9 verdict table: canonQ_ref old (dispatch bug, duplicates canon_ref) vs the
fixed dispatch (support-side orbital prototype + shared reference gauge), paired
across the 168-condition B20 grid on bit-identical episodes."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "04_results" / "logs"
TAB = BASE / "04_results" / "tables"

d = json.loads((LOGS / "B40_canonq_fix.json").read_text())
rows = d["conds"]

# by sigma_theta
by_st = {}
for r in rows:
    by_st.setdefault(round(r["sigma_th"], 3), []).append(r)

lines = [
    "# T29: canonQ_ref dispatch fix (A-J9), paired on the 168-condition grid\n",
    "Old column = the as-run B20 column (support side duplicated canon_ref -- the",
    "disclosed dispatch bug). New column = the B18-defined dispatch: support-side",
    "orbital prototype, both prototype and query canonicalized into the pooled",
    "support reference gauge. Same episodes (deterministic seed scheme), 2000 paired",
    "episodes per condition.\n",
    f"Overall: old {d['mean_old']:.2f}% vs new {d['mean_new']:.2f}%, "
    f"diff {d['mean_diff_pt']:+.2f} pt (win rate {d['win_rate_new']:.2f}, "
    f"Wilcoxon p = {d['p_wilcoxon']:.2e}, bootstrap CI "
    f"[{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]). The mass of ties sits at low",
    "sigma_theta where the two gauges coincide by construction; the gain concentrates",
    "at large sigma_theta.\n",
    "By sigma_theta (mean over conditions, %):",
    "",
    "| sigma_theta | n_cond | old (buggy) | new (fixed) | diff pt |",
    "|---|---|---|---|---|",
]
for st in sorted(by_st):
    g = by_st[st]
    o = 100 * sum(r["canon_ref"] for r in g) / len(g)
    n = 100 * sum(r["canonQ_ref_fixed"] for r in g) / len(g)
    lines.append(f"| {st:.3f} | {len(g)} | {o:.1f} | {n:.1f} | {n-o:+.1f} |")
lines.append("")
lines.append("Reading: the fixed dispatch strictly dominates the buggy one where the")
lines.append("support-side gauge choice matters (large sigma_theta), is identical where")
lines.append("it does not, and never loses on the cross-condition mean. The fixed column")
lines.append("is what the released grid reports; the old column is retained in the")
lines.append("reproducibility record for bit-identical reproduction of the original run.")
(TAB / "T29_canonq_fix.md").write_text("\n".join(lines))
print(f"old={d['mean_old']:.2f} new={d['mean_new']:.2f} diff={d['mean_diff_pt']:+.2f} "
      f"p={d['p_wilcoxon']:.2e} win={d['win_rate_new']:.2f}")
print("wrote T29_canonq_fix.md")
for st in sorted(by_st):
    g = by_st[st]
    o = 100 * sum(r["canon_ref"] for r in g) / len(g)
    n = 100 * sum(r["canonQ_ref_fixed"] for r in g) / len(g)
    print(f"  st={st:.3f}: old={o:.1f} new={n:.1f} ({n-o:+.1f})")
