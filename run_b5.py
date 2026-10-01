"""B5：合成全网格少样本实验（估计器级，无编码器）+ B3 证书统计。

网格：
  主网格 static: sigma_th(7) x rho(3) x p(2) x k(2) x fade_q(2)  [+ sigma 扫描]
  变体 slice: rho=0.3,p=64,fade=0 上 cfo_ramp / bimodal / absphase x sigma_th(7) x k(2)
方法 13 个（见 estim.run_method）。每条件 E 个 episode（chunk 250），逐episode 0/1 配对记录
（per-episode 0/1 数组仅供 chunk 内配对检验，不进落盘产物）。

假设检验：
  H1b: phasemap vs min(euclid, orbital) 的配对 Wilcoxon（5-shot 部分相干区）
  H2:  风险差曲线随 sigma_th 单调性（B9 汇总时检验）
"""
import argparse, json, os, sys, time, zlib
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E

METHODS = ["euclid", "cosine", "hermitian", "orbital", "tta_euclid", "circcoord",
           "whiten", "drop_agc", "phasemap", "phasemap_ml", "phasemap_w", "phasemap_unc",
           "phasemap_ok", "oracle"]

SIGMAS = [0.0, np.pi / 12, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]


def angle_err(mu_hat, mu):
    """sin of 商空间角误差 [mu_hat] vs [mu]。mu_hat:(E,K,p) 或 (K,p)；mu:(K,p)。"""
    ipn = np.abs(np.einsum("...p,...p->...", mu_hat.conj(), mu)) / \
        np.maximum(np.linalg.norm(mu_hat, axis=-1) *
                   np.linalg.norm(mu, axis=-1)[..., None, :] if mu_hat.ndim == 3
                   else np.linalg.norm(mu_hat, axis=-1) * np.linalg.norm(mu, axis=-1), 1e-12)
    return np.sqrt(np.maximum(0.0, 1 - np.minimum(ipn, 1.0) ** 2))


def run_condition(mu, K, k, m, sigma_th, sigma, Epi, seed, variant="static",
                  fade_q=0.0, noise_het_kappa=None, sigma_theta_hi=None,
                  kappa_meta=np.pi / 3, chunk=125, with_cert=False, with_angle=False):
    n_ok = {mth: np.zeros(Epi) for mth in METHODS}
    ang = {mth: [] for mth in ["euclid", "orbital", "phasemap"]}
    em_it, gap3, km_acc = [], [], []
    done = 0
    while done < Epi:
        n = min(chunk, Epi - done)
        Zs, Zq, yq = S.batch_episodes(mu, K, k, m, sigma_th, sigma, n, seed=seed + done,
                                      variant=variant, fade_q=fade_q,
                                      noise_het_kappa=noise_het_kappa,
                                      sigma_theta_hi=sigma_theta_hi)
        for mth in METHODS:
            st_o = sigma_th if (mth == "phasemap_ok" and variant in ("static", "bimodal", "cfo_ramp")) else None
            pred, aux = E.run_method(mth, Zs, Zq, mu_true=mu,
                                     kappa_meta=kappa_meta, sigma_th_oracle=st_o)
            n_ok[mth][done:done + n] = (pred == yq).mean(axis=1)
        if with_angle and k >= 2:
            ang["euclid"].extend(angle_err(E.proto_euclid(Zs), mu).ravel().tolist())
            ang["orbital"].extend(angle_err(E.proto_orbital(Zs), mu).ravel().tolist())
            mu_pm, _ = E.phasemap_em(Zs, kappa_meta=kappa_meta)
            ang["phasemap"].extend(angle_err(mu_pm, mu).ravel().tolist())
        if k == 1:
            for km in (1e-3, np.pi / 6, np.pi / 3, 2 * np.pi / 3):
                mu_km, aux_km = E.phasemap_em(Zs, kappa_meta=km)
                pred_km = E.cls_marginal(Zq, mu_km, aux_km)
                km_acc.append(float((pred_km == yq).mean()))
        if with_cert:
            _, aux_pm = E.phasemap_em(Zs, kappa_meta=kappa_meta)
            em_it.extend([aux_pm["iters"]] * len(Zs))
            mu3 = E.orbital_align(Zs, restarts=3)
            mu20 = E.orbital_align(Zs, restarts=20, n_iter=14)
            o3 = np.abs(np.einsum("ekp,ekjp->ekj", mu3.conj(), Zs)).sum(-1)
            o20 = np.abs(np.einsum("ekp,ekjp->ekj", mu20.conj(), Zs)).sum(-1)
            rel = np.maximum((o20 - o3) / np.maximum(o3, 1e-12), 0)
            gap3.extend(rel.ravel().tolist())
        done += n
    out = {"acc": {mth: float(np.mean(n_ok[mth])) for mth in METHODS},
           "acc_se": {mth: float(np.std(n_ok[mth]) / np.sqrt(Epi)) for mth in METHODS}}
    if k == 1 and km_acc:
        out["phasemap_kmsweep"] = km_acc
    if k >= 2 and with_angle:
        out["angle_err_median"] = {mth: float(np.median(v)) for mth, v in ang.items() if v}
    if with_cert:
        out["cert"] = {"em_iters_mean": float(np.mean(em_it)),
                       "restart_gap_p50": float(np.percentile(gap3, 50)),
                       "restart_gap_p95": float(np.percentile(gap3, 95)),
                       "gap_zero_frac": float(np.mean(np.array(gap3) < 1e-9))}
    return out, n_ok


def paired_test(a, b):
    """配对 Wilcoxon 符号秩 + 差值 bootstrap 95% CI。a,b: 0/1 数组。"""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, 0):
        return {"p": 1.0, "ci": [0.0, 0.0], "median_d": 0.0}
    try:
        p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except Exception:
        p = float("nan")
    rng = np.random.RandomState(0)
    bs = [float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(2000)]
    return {"p": p, "median_d": float(np.median(d)),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def _cond_worker(job):
    (variant, p, rho, k, st, fade_q, sig, tag, sigma_theta_hi, K, m, epi) = job
    noise_het_kappa = 10.0 if tag == "het" else None
    import zlib as _z
    key = "|".join(f"{x}={y}" for x, y in sorted(
        {"tag": tag, "p": p, "rho": rho, "k": k, "st": round(st, 4),
         "fade": fade_q, "sig": sig}.items()))
    mu = S.make_prototypes(K, p, rho, np.random.RandomState(_z.crc32(("mu:" + key).encode())))
    with_cert = (k == 5 and p == 64 and rho == 0.3 and fade_q == 0.0 and tag == "main")
    r, n_ok = run_condition(mu, K, k, m, st, sig, epi, seed=_z.crc32(key.encode()),
                            variant=variant, fade_q=fade_q, sigma_theta_hi=sigma_theta_hi,
                            noise_het_kappa=noise_het_kappa,
                            with_cert=with_cert, with_angle=(k == 5))
    r.update({"cond": key, "tag": tag, "variant": variant, "p": p, "rho": rho, "k": k,
              "sigma_th": st, "fade_q": fade_q, "sigma": sig, "K": K, "epi": epi})
    if k == 5:
        better = "euclid" if np.mean(n_ok["euclid"]) >= np.mean(n_ok["orbital"]) else "orbital"
        r["h1b_vs_best_endpoint"] = paired_test(n_ok["phasemap"], n_ok[better])
        r["h1b_best_endpoint"] = better
    r["angle_err_median"] = r.get("angle_err_median", {})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--epi", type=int, default=2000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--m", type=int, default=75)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = a.out or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    epi = 400 if a.quick else a.epi
    rows, variants_rows = [], []
    t00 = time.time()

    def cid(**kw):
        return "|".join(f"{k2}={v2}" for k2, v2 in sorted(kw.items()))

    jobs = []

    def add_job(variant, p, rho, k, st, fade_q, sig, tag="main", sigma_theta_hi=None):
        jobs.append((variant, p, rho, k, st, fade_q, sig, tag, sigma_theta_hi,
                     a.K, a.m, epi))

    # 主网格
    for fade_q in [0.0, 0.5]:
        for p in [16, 64]:
            for rho in [0.0, 0.3, 0.7]:
                for k in [1, 5]:
                    for st in SIGMAS:
                        add_job("static", p, rho, k, st, fade_q, a.sigma)
    # sigma 扫描
    for sig in [0.15, 0.6]:
        for k in [1, 5]:
            for st in SIGMAS:
                add_job("static", 64, 0.3, k, st, 0.0, sig, tag="sig")
    # 噪声异方差 slice（P3 加权主战场）
    for k in [1, 5]:
        for st in SIGMAS:
            add_job("static", 64, 0.3, k, st, 0.0, a.sigma, tag="het")
    # 变体 slice
    for variant in ["cfo_ramp", "bimodal", "absphase"]:
        for k in [1, 5]:
            for st in SIGMAS:
                add_job(variant, 64, 0.3, k, st, 0.0, a.sigma, tag="var", sigma_theta_hi=np.pi / 2)

    print(f"total conditions: {len(jobs)}, workers={min(a.workers, len(jobs))}", flush=True)
    import multiprocessing as mp
    with mp.Pool(processes=min(a.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(_cond_worker, jobs)):
            (rows if r["tag"] == "main" else variants_rows).append(r)
            if i % 10 == 0 or i == len(jobs) - 1:
                print(f"[{time.time()-t00:6.0f}s] done {i+1}/{len(jobs)}: {r['cond']} "
                      f"euclid={100*r['acc']['euclid']:5.1f} orbital={100*r['acc']['orbital']:5.1f} "
                      f"phasemap={100*r['acc']['phasemap']:5.1f} "
                      f"pm_ok={100*r['acc']['phasemap_ok']:5.1f} oracle={100*r['acc']['oracle']:5.1f}",
                      flush=True)

    rows.sort(key=lambda r: r["cond"])
    variants_rows.sort(key=lambda r: r["cond"])
    # H1b 汇总
    summary = {"H1b": []}
    for r in rows:
        if "h1b_vs_best_endpoint" in r and r.get("sigma_th") in (np.pi / 6, np.pi / 3, np.pi / 2) \
                and r.get("rho") in (0.3, 0.7) and r.get("tag") == "main" and r.get("fade_q") == 0.0:
            summary["H1b"].append({"cond": r["cond"], **r["h1b_vs_best_endpoint"],
                                   "acc_phasemap": r["acc"]["phasemap"],
                                   "acc_euclid": r["acc"]["euclid"], "acc_orbital": r["acc"]["orbital"]})
    json.dump(rows, open(os.path.join(out, "logs", "B5_grid.json"), "w"), indent=1)
    json.dump(variants_rows, open(os.path.join(out, "logs", "B5_variants.json"), "w"), indent=1)
    json.dump([r["cert"] for r in rows if "cert" in r],
              open(os.path.join(out, "logs", "B3_certificate.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out, "logs", "B5_summary_tests.json"), "w"), indent=1)
    print(f"\nB5 DONE in {(time.time()-t00)/60:.1f} min; main={len(rows)} variants={len(variants_rows)}")
    print("=== H1b（部分相干区，phasemap vs 最优端点）===")
    for h in summary["H1b"]:
        print(f"  {h['cond']}: d={100*h['median_d']:+.1f}pts p={h['p']:.2e} "
              f"ci=[{100*h['ci'][0]:+.1f},{100*h['ci'][1]:+.1f}]")


if __name__ == "__main__":
    main()
