"""B5-extra：H1b 关键条件的配对检验（phasemap_ml vs 按均值更优端点），2000 epi。

条件: sigma_th ∈ {pi/6, pi/3, pi/2} x rho ∈ {0.3,0.7} x p=64 x fade{0,0.5} x k=5
"""
import base64, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E

SIGMAS = [np.pi / 6, np.pi / 3, np.pi / 2]


def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    try:
        p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except Exception:
        p = float("nan")
    rng = np.random.RandomState(0)
    bs = [float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(2000)]
    return {"p": p, "median_d": float(np.median(d)),
            "mean_d": float(np.mean(d)),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def run_cell(K, p, rho, k, st, sig, fade_q, epi, chunk=125):
    rng = np.random.RandomState(20260929)
    mu = S.make_prototypes(K, p, rho, rng)
    n_ok = {"euclid": np.zeros(epi), "orbital": np.zeros(epi), "phasemap_ml": np.zeros(epi)}
    done = 0
    while done < epi:
        n = min(chunk, epi - done)
        Zs, Zq, yq = S.batch_episodes(mu, K, k, 75, st, sig, n, seed=777 + done, fade_q=fade_q)
        n_ok["euclid"][done:done+n] = (E.cls_euclid(Zq, E.proto_euclid(Zs)) == yq).mean(1)
        n_ok["orbital"][done:done+n] = (E.cls_orbital(Zq, E.proto_orbital(Zs)) == yq).mean(1)
        mu_pm, aux = E.phasemap_em(Zs)
        n_ok["phasemap_ml"][done:done+n] = (E.cls_marginal(Zq, mu_pm, aux) == yq).mean(1)
        done += n
    better = "euclid" if n_ok["euclid"].mean() >= n_ok["orbital"].mean() else "orbital"
    t = paired(n_ok["phasemap_ml"], n_ok[better])
    t.update({"better_endpoint": better,
              "acc_phasemap_ml": float(n_ok["phasemap_ml"].mean()),
              "acc_euclid": float(n_ok["euclid"].mean()),
              "acc_orbital": float(n_ok["orbital"].mean())})
    return t


def _worker(job):
    rho, st, fade_q = job
    r = run_cell(5, 64, rho, 5, st, 0.3, fade_q, 2000)
    r.update({"rho": rho, "sigma_th": st, "fade_q": fade_q})
    return r


if __name__ == "__main__":
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
    jobs = [(rho, st, fq) for fq in (0.0, 0.5) for rho in (0.3, 0.7) for st in SIGMAS]
    import multiprocessing as mp
    res = []
    with mp.Pool(6, initializer=None) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            res.append(r)
            print(f"rho={r['rho']} st={r['sigma_th']:.2f} fade={r['fade_q']}: "
                  f"phML={100*r['acc_phasemap_ml']:.1f} vs {r['better_endpoint']}="
                  f"{100*max(r['acc_euclid'], r['acc_orbital']):.1f} "
                  f"p={r['p']:.1e} ci=[{100*r['ci'][0]:+.2f},{100*r['ci'][1]:+.2f}]", flush=True)
    res.sort(key=lambda r: (r["fade_q"], r["rho"], r["sigma_th"]))
    json.dump(res, open(os.path.join(out, "logs", "B5_extra_h1b_ml.json"), "w"), indent=1)
    print("B5-extra DONE")
