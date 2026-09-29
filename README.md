# CVFEW: Few-Shot Learning on Complex Representations Modulo Global Phase

PhaseMAP: phase-marginalized adaptive prototypes for few-shot learning on
complex-valued observations with per-sample global-phase nuisance
(research direction 8 of the CVCNN roadmap).

## Key result
PhaseMAP-ML (EM with Bessel-closed-form / grid E-step + full marginal-likelihood
classifier) matches the best of {Euclidean prototype, hard-orbital prototype}
within 0.5 pt at both sigma_theta -> 0 and sigma_theta -> pi, and BEATS both
fixed endpoints in the partial-coherence region (sigma_theta = pi/3:
+1.76 pt, paired Wilcoxon p = 1.1e-156, 2000 paired episodes), while the
endpoint envelope collapses by up to 58 pt across the sigma_theta axis.

## Layout
- cvfe/synth.py     PhaseFewSyn controllable synthetic episode family
- cvfe/estim.py     estimator suite: euclid/cosine/hermitian/orbital(=complex L1-PCA
                    coordinate ascent)/TTA/per-coordinate circular mean/whitening/
                    drop+AGC engineering baseline/PhaseMAP EM (Bessel + grid gamma,
                    pooled debiased kappa, heteroscedastic weights, uncertainty
                    correction) + full marginal-likelihood classifier
- cvfe/episodes.py  RadioML RML2016.10a few-shot episode sampler (SNR-stratified,
                    phase-injection suite: random global phase / Wiener phase noise /
                    CFO ramp)
- cvfe/torch_nets.py equivariant complex-conv trunk (delta = 1e-4) + real dual-channel
                    control trunk with exact independent-real-scalar parameter counts
- run_b1.py         real-IQ phase-structure diagnostic + quotient-premise gate
- run_b4.py         theory-numerics: P4 risk separation (analytic = MC to 2e-4),
                    P3 weighting dichotomy, P2(iii) degeneracy
- run_b5.py         synthetic full grid (252 conditions x 2000 paired episodes)
- run_b5_extra.py   H1b paired tests
- run_b8.py         real-IQ few-shot (5-way, SNR-stratified, injection grid)
- run_b67.py        equivariance diagnostic, frozen-encoder few-shot, end-to-end
                    orbital-vs-augmentation budget study (server, torch)
- make_report_b9.py figures + tables

## Reproduce
python cvfe/tests_self.py          # 15/15 unit tests (closed forms, endpoints)
python run_b5.py --quick           # smoke grid
python run_b5.py --epi 2000        # full synthetic grid
python run_b8.py --epi 600         # real-IQ few-shot
