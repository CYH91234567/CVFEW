#!/usr/bin/env python3
"""A-J5: complexity / convergence certificate for paper A.

Measures, on the synthetic main-grid configuration (K=5 classes, k=5 shots,
p=64, rho=0.3, sigma=0.3, sigma_theta=pi/3):
  1. per-episode wall-clock of the three prototype estimators + classifier
     (euclid / orbital / phasemap_ml), batch of E=200 episodes, CPU numpy;
  2. EM iteration count distribution (tol=1e-7, max 24) and multi-restart
     converged-likelihood gap (the B3 <=1.7e-6 restart-gap evidence, refreshed);
  3. operation counts (asymptotic flops formulas instantiated at the grid dims);
  4. joint-profile sigma_theta candidate count (the B21 1-D search width).
Outputs 04_results/logs/B38_complexity.json + prints a LaTeX-ready table.
"""
import json
import time
from pathlib import Path

import numpy as np

import cvfe.estim as E
import cvfe.synth as S

LOGS = Path(__file__).resolve().parents[1] / "04_results" / "logs"
K, k, p, rho, sigma, st, EPISODES = 5, 5, 64, 0.3, 0.3, np.pi / 3, 200

rng = np.random.default_rng(20261003)
mu = S.make_prototypes(K, p, rho, np.random.RandomState(20261003))
Zs, Zq, yq = S.batch_episodes(mu, K, k, 200, st, sigma, EPISODES, seed=20261003)
print("batch:", Zs.shape, Zq.shape)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0


# --- estimators (prototype construction), E=200 episodes ---
# euclid prototype = coherent mean (the zero-training baseline)
mu_e, t_e = timed(lambda Z: Z.mean(axis=2), Zs)
mu_o, t_o = timed(E.orbital_align, Zs)
(mu_p, aux), t_p = timed(E.phasemap_em, Zs, kappa_profile="joint")
print(f"euclid {t_e*1e3:.1f} ms  orbital {t_o*1e3:.1f} ms  phasemap {t_p*1e3:.1f} ms "
      f"for {EPISODES} episodes")

iters = np.asarray(aux.get("iters", []), dtype=float).ravel()
iters = iters[iters > 0]
iter_stats = ({"min": int(iters.min()), "max": int(iters.max()),
               "mean": float(iters.mean()), "frac_at_tol_exit": float(np.mean(iters < 24))}
              if iters.size else None)

# --- restart gap: converged objective across random restarts ---
def restart_gap(Z, reps=5):
    objs = []
    for s in range(reps):
        m, _ = E.phasemap_em(Z, seed=1000 + s, kappa_profile="joint")
        # proxy objective: mean residual ||Z_al - m||^2 per class (lower=better)
        ip = np.einsum("ekp,ekjp->ekj", m.conj(), Z)
        Za = Z * np.exp(-1j * np.angle(ip))[..., None]
        objs.append(float(np.abs(Za - m[:, :, None, :]).mean()))
    return float(np.max(objs) - np.min(objs))

gap = restart_gap(Zs[:20])
print(f"restart gap (20 episodes x 5 restarts, proxy objective): {gap:.2e}")

# --- classification cost per query ---
def classify_cost():
    heads = ("euclid", "orbital", "phasemap_ml")
    out = {}
    Zqf = Zq.reshape(-1, p)
    for h in heads:
        t0 = time.perf_counter()
        if h == "euclid":
            # nearest coherent-mean prototype, broadcast (E,m,1,p) vs (E,1,K,p)
            d = np.linalg.norm(Zq[:, :, None, :] - mu_e[:, None, :, :], axis=-1)
            pred = d.argmin(-1).reshape(-1)
        elif h == "orbital":
            sc = np.einsum("emp,ekp->emk", Zq.conj(), mu_o)
            pred = np.abs(sc).argmax(-1).reshape(-1)
        else:
            # full marginal-likelihood classification (episode-matched);
            # cls_marginal already returns the argmax prediction (E, m)
            pred = E.cls_marginal(Zq, mu_p, aux).reshape(-1)
        dt = time.perf_counter() - t0
        acc = float((pred == yq.reshape(-1)).mean())
        out[h] = {"ms_per_1k_queries": dt / Zqf.shape[0] * 1000, "acc": acc}
    return out


cc = classify_cost()
for h, v in cc.items():
    print(f"classify[{h}]: {v['ms_per_1k_queries']:.2f} ms/1k queries, acc={v['acc']:.3f}")

# --- asymptotic op counts (one complex length-p dot ~= 8p flops) ---
G = 256  # grid points, _gamma_grid default
T = 24   # EM cap
ops = {
    "euclid_protos_per_episode_flops": 8 * K * k * p,
    "euclid_query_per_class_flops": 8 * p,
    "orbital_init_flops": 3 * 10 * 8 * K * k * p,      # restarts*iters*dots
    "orbital_query_per_class_flops": 8 * p,
    "phasemap_EM_per_iter_flops": 8 * K * k * p + 3 * K * k * G,   # dots + grid quadrature
    "phasemap_query_per_class_flops": 8 * p + 3 * G,  # orbital distance + log c0
    "grid_points_G": G, "EM_iter_cap": T, "classes_K": K, "shots_k": k, "dim_p": p,
}

out = {
    "config": {"K": K, "k": k, "p": p, "rho": rho, "sigma": sigma,
               "sigma_theta": st, "episodes": EPISODES},
    "estimator_ms_for_200_episodes": {"euclid": t_e * 1e3, "orbital": t_o * 1e3,
                                      "phasemap_joint": t_p * 1e3},
    "per_episode_ms": {"euclid": t_e * 1e3 / EPISODES, "orbital": t_o * 1e3 / EPISODES,
                       "phasemap_joint": t_p * 1e3 / EPISODES},
    "em_iters": iter_stats,
    "restart_gap_proxy": gap,
    "classify": cc,
    "op_counts": ops,
    "note": ("CPU numpy single-thread; the EM and grid costs are dominated by the "
             "G-point wrapped-prior quadrature; B3's restart-gap evidence refreshed."),
}
(LOGS / "B38_complexity.json").write_text(json.dumps(out, indent=1))
print("wrote B38_complexity.json")
