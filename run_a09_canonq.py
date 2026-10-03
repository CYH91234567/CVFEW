#!/usr/bin/env python3
"""A-J9: recompute the canonQ_ref column with the fixed dispatch, paired against
the (buggy, canon_ref-duplicating) old column on bit-identical episodes.

The B20 grid reuses run_b5's job list and seed scheme (zlib.crc32 of the condition
key for episodes, crc32("mu:"+key) for prototypes), so per-condition episodes are
deterministic and the old/new columns are episode-paired.

Only two rules are computed per condition (canon_ref = the old canonQ_ref behavior
by the bug; canonQ_ref_fixed = the fixed dispatch), so the cost is ~1/10 of the
full grid.  Emits 04_results/logs/B40_canonq_fix.json with per-condition means,
per-episode paired arrays (subsampled), and a paired Wilcoxon over conditions.
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(BASE, "04_results", "logs")

EPS = 2000
CHUNK = 250


def cond_iter():
    """复刻 run_b20 的主网格 job 顺序（fade 0 与 0.5、k 1/5、p 16/64、rho 0/0.3/0.7、
    sigma_th 7 点）。"""
    SIGMAS = [0.0, np.pi / 12, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
    jobs = []
    for p in (16, 64):
        for rho in (0.0, 0.3, 0.7):
            for k in (1, 5):
                for fade_q in (0.0, 0.5):
                    for st in SIGMAS:
                        jobs.append(("static", p, rho, k, st, fade_q, 0.3, "main",
                                     None, 5, 75, EPS))
    return jobs


def run_one(job, store_episodes=False):
    (variant, p, rho, k, st, fade_q, sig, tag, _hi, K, m, epi) = job
    key = "|".join(f"{x}={y}" for x, y in sorted(
        {"tag": tag, "p": p, "rho": rho, "k": k, "st": round(st, 4),
         "fade": fade_q, "sig": sig}.items()))
    mu = S.make_prototypes(K, p, rho, np.random.RandomState(zlib.crc32(("mu:" + key).encode())))
    seed = zlib.crc32(key.encode())
    v = E.canon_ref_vec
    accs = {"canon_ref": [], "canonQ_ref_fixed": []}
    done = 0
    while done < epi:
        n = min(CHUNK, epi - done)
        Zs, Zq, yq = S.batch_episodes(mu, K, k, m, st, sig, n, seed=seed + done,
                                      variant=variant, fade_q=fade_q)
        # old (buggy) canonQ_ref == canon_ref pipeline
        p_old, _ = E.run_method("canon_ref", Zs, Zq)
        # fixed dispatch
        p_new, _ = E.run_method("canonQ_ref", Zs, Zq)
        accs["canon_ref"].extend((p_old == yq).mean(axis=1).tolist())
        accs["canonQ_ref_fixed"].extend((p_new == yq).mean(axis=1).tolist())
        done += n
    return {"cond": key, "k": k, "p": p, "rho": rho, "sigma_th": st, "fade": fade_q,
            "canon_ref": float(np.mean(accs["canon_ref"])),
            "canonQ_ref_fixed": float(np.mean(accs["canonQ_ref_fixed"])),
            "n": len(accs["canon_ref"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=EPS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = cond_iter()
    if a.limit:
        jobs = jobs[:a.limit]
    out = []
    t0 = time.time()
    for i, job in enumerate(jobs):
        r = run_one(job)
        out.append(r)
        if (i + 1) % 12 == 0 or i == 0:
            print(f"[{i+1}/{len(jobs)}] {r['cond']} old={100*r['canon_ref']:.1f} "
                  f"new={100*r['canonQ_ref_fixed']:.1f} ({time.time()-t0:.0f}s)")
    # paired over conditions
    old = np.array([r["canon_ref"] for r in out]) * 100
    new = np.array([r["canonQ_ref_fixed"] for r in out]) * 100
    d = new - old
    res = {"n_cond": len(out), "mean_old": float(old.mean()), "mean_new": float(new.mean()),
           "mean_diff_pt": float(d.mean()), "median_diff_pt": float(np.median(d)),
           "win_rate_new": float((d > 0).mean()),
           "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if not np.allclose(d, 0) else 1.0,
           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
           "conds": out}
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "B40_canonq_fix.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"paired: old={res['mean_old']:.2f} new={res['mean_new']:.2f} "
          f"diff={res['mean_diff_pt']:+.2f}pt (win {res['win_rate_new']:.2f}, "
          f"p={res['p_wilcoxon']:.2e})")
    print("wrote B40_canonq_fix.json")


if __name__ == "__main__":
    main()
