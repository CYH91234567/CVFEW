"""B20：canon 系基线并入合成主网格（PREREG_B20）。

复用 B5 的主网格与运行器，方法表扩至 19（+canon_max/canon_wmean/canon_ref/
canonX_ref/canonQ_ref）。输出 B20_grid.json（不覆盖 B5_grid.json）。
κ 估计用 B21 最终默认 profile（版本注记写入表）。
"""
import argparse, json, os, sys, time, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_b5 as R
from cvfe import synth as S

R.METHODS = R.METHODS + ["canon_first", "canon_mean", "canon_max", "canon_wmean",
                         "canon_ref", "canonX_ref", "canonQ_ref"]


def _cond_worker(job):
    (variant, p, rho, k, st, fade_q, sig, tag, sigma_theta_hi, K, m, epi) = job
    key = "|".join(f"{x}={y}" for x, y in sorted(
        {"tag": tag, "p": p, "rho": rho, "k": k, "st": round(st, 4),
         "fade": fade_q, "sig": sig}.items()))
    mu = S.make_prototypes(K, p, rho, np.random.RandomState(zlib.crc32(("mu:" + key).encode())))
    r, n_ok = R.run_condition(mu, K, k, m, st, sig, epi, seed=zlib.crc32(key.encode()),
                              variant=variant, fade_q=fade_q, sigma_theta_hi=sigma_theta_hi)
    r.update({"cond": key, "tag": tag, "variant": variant, "p": p, "rho": rho, "k": k,
              "sigma_th": st, "fade_q": fade_q, "sigma": sig, "K": K, "epi": epi})
    if k == 5:
        better = "euclid" if np.mean(n_ok["euclid"]) >= np.mean(n_ok["orbital"]) else "orbital"
        r["h1b_vs_best_endpoint"] = R.paired_test(n_ok["phasemap_ml"], n_ok[better])
        r["h1b_best_endpoint"] = better
        # P7 判定用配对：canon_ref vs orbital / canon_max vs canon_ref
        r["p7_canonref_vs_orbital"] = R.paired_test(n_ok["canon_ref"], n_ok["orbital"])
        r["p7_canonmax_vs_canonref"] = R.paired_test(n_ok["canon_max"], n_ok["canon_ref"])
    r.pop("per_episode_b64", None)
    return r


def _cond_key(job):
    (variant, p, rho, k, st, fade_q, sig, tag, _hi, K, m, epi) = job
    return "|".join(f"{x}={y}" for x, y in sorted(
        {"tag": tag, "p": p, "rho": rho, "k": k, "st": round(st, 4),
         "fade": fade_q, "sig": sig}.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--epi", type=int, default=2000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--m", type=int, default=75)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = a.out or base
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    epi = 400 if a.quick else a.epi
    t00 = time.time()
    jobs = []
    for fade_q in [0.0, 0.5]:
        for p in [16, 64]:
            for rho in [0.0, 0.3, 0.7]:
                for k in [1, 5]:
                    for st in R.SIGMAS:
                        jobs.append(("static", p, rho, k, st, fade_q, a.sigma, "main", None,
                                     a.K, a.m, epi))
    part_path = os.path.join(out, "logs", "B20_grid.json")
    rows = []
    if os.path.exists(part_path):                          # 断点续跑（方法不全的记录作废重算）
        try:
            prev = json.load(open(part_path))
            prev_recs = prev["records"] if isinstance(prev, dict) else prev
            full = set(R.METHODS)
            rows = [r for r in prev_recs if set(r["acc"]) == full]
            done = {_cond_key(j) for j in jobs} & {r["cond"] for r in rows}
            jobs = [j for j in jobs if _cond_key(j) not in done]
            print(f"resume: {len(rows)} complete ({len(prev_recs)} on file), "
                  f"{len(jobs)} to go", flush=True)
        except Exception as e:
            print("resume load failed:", e, flush=True)
    print(f"B20 total conditions: {len(jobs)}, methods={len(R.METHODS)}, "
          f"workers={min(a.workers, len(jobs))}", flush=True)
    import multiprocessing as mp
    with mp.Pool(processes=min(a.workers, max(len(jobs), 1))) as pool:
        for i, r in enumerate(pool.imap_unordered(_cond_worker, jobs)):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs) - 1:
                print(f"[{time.time()-t00:6.0f}s] done {i+1}/{len(jobs)}: {r['cond']} "
                      f"euclid={100*r['acc']['euclid']:5.1f} orbital={100*r['acc']['orbital']:5.1f} "
                      f"canon_ref={100*r['acc']['canon_ref']:5.1f} "
                      f"phML={100*r['acc']['phasemap_ml']:5.1f}", flush=True)
            if (i + 1) % 5 == 0:                           # 每5条增量落盘（断点保护）
                try:
                    rows.sort(key=lambda r: r["cond"])
                    json.dump({"methods": R.METHODS, "records": rows},
                              open(part_path, "w"), indent=1)
                except Exception as e:
                    print("dump failed:", e, flush=True)
    rows.sort(key=lambda r: r["cond"])
    json.dump({"methods": R.METHODS,
               "kappa_profile": "B21-final-default(见 estim.phasemap_em 签名)",
               "records": rows},
              open(part_path, "w"), indent=1)
    print(f"\nB20 DONE in {(time.time()-t00)/60:.1f} min; {len(rows)} conditions -> B20_grid.json")


if __name__ == "__main__":
    main()
