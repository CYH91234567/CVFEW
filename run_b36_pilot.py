#!/usr/bin/env python3
"""B36: pilot-residual interface experiment (gap-tsp-tnnls A-J1 core; Thm 3 corollary).

Theory target. Thm 3 says: with the phase known, the Euclidean rule is Bayes, so the
quotient machinery adds nothing; its corollary is that the *value* of the quotient
route is bounded by the phase-estimation error. The interface itself was never
measured -- this script measures it on the synthetic grid + (deterministic) real IQ.

Model. The receiver is pilot-aided: for every frame it holds an estimate
theta_hat_j = theta_j + delta_j with residual delta_j ~ wrapN(0, sigma_res^2)
(sigma_res = 0 -> perfect pilot; sigma_res = pi -> pilot useless). Three ways to use
the pilot:
  (a) SELECT: de-rotate z_j by theta_hat_j and classify with the Euclidean rule
      (support and query side both de-rotated). The textbook baseline.
  (b) MARGINALIZE: de-rotate, then run PhaseMAP with the phase prior width fixed to
      the *known* residual sigma_res (z_j e^{-i theta_hat_j} = e^{i delta_j} mu + eps
      is exactly the wrapN(0, sigma_res) model). Bayesian-correct integration.
  (c) IGNORE: orbital / PhaseMAP with episode-estimated sigma_theta, no pilot.
Prediction (Thm-3 corollary, quantitative): the advantage of (c) over (a) shrinks
monotonically to 0 as sigma_res -> 0; (b) never loses to (a) because it marginalizes
the residual instead of selecting a point estimate; and with oracle prototypes the
de-rotated Euclidean rule at residual sigma_res *is* the Euclidean rule at phase
width sigma_res, so the pilot-residual axis coincides with the sigma_theta axis and
the crossing sigma_res* = sigma_theta* (SC-2, the Thm 4 bridge).

Sweep sigma_res in {0, .05, .1, .2, .3, pi/12, pi/6, pi/3, pi}. Grid: K=5, k=5,
p=64, rho=0.3, sigma=0.3, fade=0; sigma_theta = pi (worst case) and pi/3 (mid).
500 episodes x 2 seeds (1000 paired episodes), 100 queries/episode.

Criteria (preregistered, PREREG_B36):
  SC-1 (value bound): max(no-pilot arms) - euclid_derot -> 0 as sigma_res -> 0.
  SC-2 (axis bridge): euclid_derot_oracle(sigma_res) == euclid_oracle(sigma_res)
      within 2 MC standard errors.
  SC-3 (marginalize the residual): phasemap_pilot >= euclid_derot across the axis,
      strictly positive in the mid band (sigma_res in [pi/12, pi/3]).
  SC-4 (crossing): the observed crossing of euclid_derot vs orbital lies in the Thm 4
      bisection bracket on the same (rho, sigma).

Outputs: 04_results/logs/B36_pilot.json (+ verdict in analyze_b36.py).
"""
import json
import numpy as np
from pathlib import Path

import cvfe.estim as E
import cvfe.synth as S

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "04_results" / "logs"

K, k, p, rho, sigma = 5, 5, 64, 0.3, 0.3
SIGMA_RES = [0.0, 0.05, 0.1, 0.2, 0.3, np.pi / 12, np.pi / 6, np.pi / 3, np.pi]
EPI, M, NSEED = 500, 100, 2


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def main():
    out = {}
    for st_query in (np.pi, np.pi / 3):
        for si, seed0 in enumerate((11, 23)):
            rng = np.random.RandomState(seed0)
            mu = S.make_prototypes(K, p, rho, np.random.RandomState(seed0 + 1))
            th_s = wrap(rng.normal(0, st_query, (EPI, K, k, 1)))
            th_q = wrap(rng.normal(0, st_query, (EPI, M, 1)))
            eps_s = (rng.randn(EPI, K, k, p) + 1j * rng.randn(EPI, K, k, p)) * (sigma / np.sqrt(2))
            eps_q = (rng.randn(EPI, M, p) + 1j * rng.randn(EPI, M, p)) * (sigma / np.sqrt(2))
            yq = np.tile(np.arange(K), (EPI, M // K)).astype(int)
            Zs = np.exp(1j * th_s) * mu[None, :, None, :] + eps_s          # (E,K,k,p)
            Zq = np.exp(1j * th_q[..., :1]) * mu[yq] + eps_q               # (E,M,p)
            mu_b = np.broadcast_to(mu, (EPI, K, p))
            # no-pilot reference curves (same episodes; arm (c))
            pred_o = E.run_method("orbital", Zs, Zq)[0]
            pred_m = E.run_method("phasemap_ml", Zs, Zq)[0]
            a_orb = float((pred_o == yq).mean())
            a_ml = float((pred_m == yq).mean())
            for sr in SIGMA_RES:
                d_s = wrap(rng.normal(0, sr, th_s.shape)) if sr > 0 else np.zeros_like(th_s)
                d_q = wrap(rng.normal(0, sr, th_q.shape)) if sr > 0 else np.zeros_like(th_q)
                # (a) SELECT: de-rotate both sides, coherent mean, Euclidean
                Zs_dr = Zs * np.exp(-1j * (th_s + d_s))
                Zq_dr = Zq * np.exp(-1j * (th_q + d_q))
                pred = E.run_method("euclid", Zs_dr, Zq_dr)[0]
                a_sel = float((pred == yq).mean())
                # oracle-prototype variants (SC-2): de-rotated query vs true prototypes,
                # and the no-pilot Euclidean at width sigma_res for the axis reference
                d2 = (np.abs(Zq_dr[:, :, None, :] - mu_b[:, None, :, :]) ** 2).sum(-1)
                a_sel_orc = float((d2.argmin(-1) == yq).mean())
                Zq_sr = np.exp(1j * wrap(rng.normal(0, sr, (EPI, M, 1)))) * mu[yq] + \
                    (rng.randn(EPI, M, p) + 1j * rng.randn(EPI, M, p)) * (sigma / np.sqrt(2))
                d2b = (np.abs(Zq_sr[:, :, None, :] - mu_b[:, None, :, :]) ** 2).sum(-1)
                a_eucl_sr = float((d2b.argmin(-1) == yq).mean())
                # (b) MARGINALIZE: de-rotate + PhaseMAP with prior width = residual
                if sr > 1e-6:
                    mu_p, aux = E.phasemap_em(Zs_dr, sigma_th=sr)
                    pred_b = E.cls_marginal(Zq_dr, mu_p, aux)
                else:  # sigma_res = 0: the exact limit of the marginal model is Euclid
                    pred_b = pred
                a_marg = float((pred_b == yq).mean())
                key = f"st{st_query:.3f}_sr{sr:.4f}_s{seed0}"
                out[key] = {"sigma_theta": st_query, "sigma_res": sr, "seed": seed0,
                            "euclid_derot": a_sel, "euclid_derot_oracle": a_sel_orc,
                            "euclid_oracle_sr": a_eucl_sr, "phasemap_pilot": a_marg,
                            "orbital": a_orb, "phasemap_ml": a_ml}
                print(f"st={st_query:.2f} sr={sr:.3f} s={si}: " + " ".join(
                    f"{nm}={100*v:.1f}" for nm, v in
                    [("sel", a_sel), ("selO", a_sel_orc), ("ref", a_eucl_sr),
                     ("marg", a_marg), ("orb", a_orb), ("ml", a_ml)]))
    (LOGS / "B36_pilot.json").write_text(json.dumps(out, indent=1))
    print("wrote", LOGS / "B36_pilot.json")


if __name__ == "__main__":
    main()
