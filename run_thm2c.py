#!/usr/bin/env python3
"""A-J3: numerical certificate for the multiclass (C-way) uniform-phase theorem.

Theory (see theory_P4_ordering.md, new Sec. "Multiclass bridge"):
  Model: K classes, unit-norm equi-coherent prototypes (mu_a^H mu_c = rho, real),
  query z = e^{i theta} mu_a + eta, eta ~ CN(0, sigma^2 I_p), equal priors.
  Euclidean rule: c_hat = argmax_c Re(z^H mu_c)  (= nearest prototype).

Results verified here:
  (1) First-order collapse: E[Re(z^H mu_c) | class a] = 0 for every (a, c)
      when theta ~ Uniform — the whole coherent signal vanishes, any K.
  (2) Pairwise exactness: K = 2 gives accuracy exactly 1/2 (Thm 2).
  (3) Multiclass formula: correct iff max_{c != a} u_c < (1 - rho) cos(theta)
      with u_c = n_c - n_a exchangeable Gaussian; hence
      P_acc = E_theta[ F( sqrt(1-rho) cos(theta) / sigma ) ],  F = CDF of max w,
      and P_acc = 1/2 * (1 - E_theta P_straddle) <= 1/2, strictly < 1/2 for K>=3.
  (4) Grid check: the formula predicts the observed 5-way Euclidean accuracy
      under uniform phase; compare with the main-grid empirical value.
  (5) Crossing inequality: P_orb < P_acc^eucl(uniform) holds on the grid
      (orbital is phase-distribution-free, Thm 1).
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm

RNG = np.random.default_rng(20261003)
LOGS = Path(__file__).resolve().parents[1] / "04_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def make_prototypes(K, p, rho, rng):
    Q = rng.standard_normal((K + 1, p)) + 1j * rng.standard_normal((K + 1, p))
    q, _ = np.linalg.qr(Q.T)
    q = q.T
    mu = np.sqrt(1 - rho) * q[1:] + np.sqrt(rho) * q[0][None, :]
    return mu.astype(np.complex128)


def euclid_acc(K, p, rho, sigma, theta_mode, n_query=400000, seed=0):
    """MC accuracy of the Euclidean nearest-prototype rule under theta_mode.

    theta_mode: 'uniform' (true Uniform(-pi, pi]) or 'wrapN_pi' (wrapped normal
    at sigma_theta = pi, the main-grid endpoint).
    """
    rng = np.random.default_rng(seed)
    mu = make_prototypes(K, p, rho, np.random.default_rng(seed + 1))
    G = np.angle(mu[:1])  # placeholder, unused
    # query classes
    y = rng.integers(0, K, n_query)
    if theta_mode == "uniform":
        th = rng.uniform(-np.pi, np.pi, n_query)
    else:  # wrapN(pi)
        th = (rng.normal(0, np.pi, n_query) + np.pi) % (2 * np.pi) - np.pi
    sig = rng.standard_normal((n_query, p)) + 1j * rng.standard_normal((n_query, p))
    eta = sig * (sigma / np.sqrt(2))
    z = np.exp(1j * th)[:, None] * mu[y] + eta
    # scores: Re(z^H mu_c) for all c  -> argmax
    S = np.real(z @ mu.conj().T)
    pred = S.argmax(1)
    acc = (pred == y).mean()
    # also first-order moment check: mean score per (true class, scored class)
    n_pairs = min(n_query, 200000)
    sub = slice(0, n_pairs)
    mom = np.zeros((K, K))
    for a in range(K):
        for c in range(K):
            m = y[sub] == a
            mom[a, c] = np.mean(S[sub][m, c])
    return acc, mom, mu


def analytic_acc(K, rho, sigma, n_grid=20001, n_mc=200000, seed=0):
    """P_acc = E_theta[ F( sqrt(1-rho) cos(theta)/sigma ) ] where F is the CDF
    of the max of (K-1) exchangeable Gaussians with unit variance and mutual
    correlation 1/2 (the exact covariance of u_c = n_c - n_a up to the common
    sigma*sqrt(1-rho) scale).  F is evaluated by MC with shared correlation.
    """
    rng = np.random.default_rng(seed)
    C = K - 1
    # u = sigma*sqrt(1-rho) * w, w ~ N(0, Sigma), Sigma_cc=1, Sigma_cc'=1/2.
    # Represent w = sqrt(1/2) e + sqrt(1/2) g_c  with e, g_c iid N(0,1):
    # Var(w_c) = 1/2 + 1/2 = 1, Cov = 1/2.  -> max_c w_c = sqrt(1/2) e + sqrt(1/2) max g_c
    # only correct in distribution if we need the MAX of correlated normals;
    # the affine representation is exact for the joint, so:
    n = n_mc
    e = rng.standard_normal(n)
    Gm = rng.standard_normal((n, C))
    wmax = np.sqrt(0.5) * e + np.sqrt(0.5) * Gm.max(1)
    # theta grid (uniform) and threshold
    th = np.linspace(-np.pi, np.pi, n_grid)
    t = np.sqrt(1 - rho) * np.cos(th) / sigma
    # F(t) = P(wmax < t): empirical CDF
    wmax_sorted = np.sort(wmax)
    F = np.searchsorted(wmax_sorted, t, side="right") / n
    acc = F.mean()
    # straddle identity check: F(t)+F(-t) = 1 - P_straddle
    Fneg = np.searchsorted(wmax_sorted, -t, side="right") / n
    straddle = 1.0 - (F + Fneg)
    return acc, straddle.mean(), (straddle > 1e-6).mean()


def orbital_acc(K, p, rho, sigma, n_query=200000, seed=0):
    """Orbital rule argmax_c |z^H mu_c| under uniform theta (Thm 1: phase-free)."""
    rng = np.random.default_rng(seed)
    mu = make_prototypes(K, p, rho, np.random.default_rng(seed + 1))
    y = rng.integers(0, K, n_query)
    th = rng.uniform(-np.pi, np.pi, n_query)
    eta = (rng.standard_normal((n_query, p)) + 1j * rng.standard_normal((n_query, p))) * (sigma / np.sqrt(2))
    z = np.exp(1j * th)[:, None] * mu[y] + eta
    S = np.abs(z @ mu.conj().T)
    return (S.argmax(1) == y).mean()


out = {}
# (1)+(2): first-order moment collapse and pairwise exactness
for K in (2, 5):
    acc, mom, mu = euclid_acc(K, 16 if K == 5 else 8, 0.3, 0.3, "uniform", seed=11)
    out[f"K{K}_uniform_acc"] = acc
    out[f"K{K}_moment_matrix_mean_abs"] = float(np.abs(mom).mean())
    out[f"K{K}_moment_matrix_max_abs"] = float(np.abs(mom).max())
    print(f"K={K} uniform-phase Euclid acc = {acc:.4f} (chance {1/K:.3f}); "
          f"mean|E[score_c|a]| = {np.abs(mom).mean():.5f}")

# (3) analytic formula + straddle identity, K=5
acc_an, straddle_mean, straddle_frac = analytic_acc(5, 0.3, 0.3)
out["K5_analytic_acc"] = acc_an
out["K5_straddle_mean"] = straddle_mean
out["K5_straddle_frac_nonzero"] = straddle_frac
print(f"K=5 analytic formula acc = {acc_an:.4f}; straddle mean = {straddle_mean:.4f}")

# (4) compare with wrapN(pi) endpoint as run in the main grid
acc_pi, _, _ = euclid_acc(5, 16, 0.3, 0.3, "wrapN_pi", n_query=200000, seed=12)
out["K5_wrapNpi_acc"] = acc_pi
print(f"K=5 wrapN(pi) endpoint acc = {acc_pi:.4f} (main-grid empirical 31.5)")

# (5) crossing inequality on the grid parameters
# Thm 4-K crossing condition: P_acc^eucl(uniform) < P_acc^orb (orbital better at
# uniform phase), together with Euclid = Bayes at delta_0 (Thm 3) and phase-path
# continuity, implies an orbital-better crossing exists for K classes.
orb = orbital_acc(5, 16, 0.3, 0.3)
out["K5_orbital_uniform_acc"] = orb
cross = out["K5_uniform_acc"] < orb
out["crossing_condition_holds"] = bool(cross)
print(f"K=5 orbital acc (uniform) = {orb:.4f}; crossing condition "
      f"P_eucl(unif) < P_orb: {cross}")

# rho sweep of the analytic formula (K=5, sigma=0.3): structure of the residual
sweep = {}
for rho in (0.0, 0.15, 0.3, 0.5, 0.7, 0.9):
    a, s, _ = analytic_acc(5, rho, 0.3, n_mc=100000, seed=7)
    sweep[rho] = {"uniform_euclid_acc": a, "straddle_mean": s}
out["rho_sweep_K5"] = sweep
print("rho sweep (K=5, sigma=0.3):", {k: round(v["uniform_euclid_acc"], 4) for k, v in sweep.items()})

# also verify Thm-1 phase-freeness numerically: orbital acc at sigma_theta = pi/6
acc_orb_pi6 = orbital_acc(5, 16, 0.3, 0.3, ) # uniform-based; re-run wrapN below
rng = np.random.default_rng(5)
mu = make_prototypes(5, 16, 0.3, np.random.default_rng(6))
y = rng.integers(0, 5, 200000)
th = (rng.normal(0, np.pi / 6, 200000) + np.pi) % (2 * np.pi) - np.pi
eta = (rng.standard_normal((200000, 16)) + 1j * rng.standard_normal((200000, 16))) * (0.3 / np.sqrt(2))
z = np.exp(1j * th)[:, None] * mu[y] + eta
S = np.abs(z @ mu.conj().T)
orb_pi6 = (S.argmax(1) == y).mean()
out["K5_orbital_wrapN_pi6_acc"] = orb_pi6
print(f"K=5 orbital acc (wrapN pi/6) = {orb_pi6:.4f} vs uniform {orb:.4f} (Thm 1 flatness)")

out["params"] = {"K": 5, "rho": 0.3, "sigma": 0.3, "p": 16,
                 "empirical_main_grid_euclid_at_sigma_pi": 0.315}
(LOGS / "B37_thm2c_multiclass.json").write_text(json.dumps(out, indent=1))
print("wrote B37_thm2c_multiclass.json")
