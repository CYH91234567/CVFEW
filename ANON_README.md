# Few-Shot Learning on Complex Representations Modulo Global Phase

Anonymous code release for the manuscript(s) on few-shot learning over
complex-valued (I/Q) observations with a per-sample global-phase nuisance.

The release contains the full experiment suite: estimation-theory experiments
(closed-form risk certificates, 252-condition synthetic grid, pilot-residual
interface, physical-channel second domain), the equivariant-complex /
phase-invariant-readout learning experiments (four-arm panels on two real
datasets), and the certificate tooling (float-level invariance certificates,
quantitative equivariance verification, audit/repair chain).

## Layout

- `code/cvfe/` — core library: synthetic episode family, estimator suite
  (Euclidean/orbital/PhaseMAP EM/marginal-likelihood), RadioML episode sampler
  with phase-injection suite, complex equivariant nets + invariant statistics,
  self-tests.
- `code/run_*.py` — experiment runners (deterministic estimation experiments
  run locally; torch-based learning experiments run on a GPU server).
- `code/analyze_*.py`, `code/make_table*.py`, `code/make_t_*.py` — verdict
  scripts that turn run products into the tables and certificate JSONs cited
  by the manuscript.
- `code/make_cache*.py` — dataset cache builders (public mirrors; see below).
- `prereg/` — pre-registration documents for the torch-based experiments
  (criteria fixed before execution; failure branches included).
- `code/cvfe/tests_self.py` — 30/30 unit tests (closed forms, endpoints,
  certificate arithmetic).

## Datasets

- RML2016.10a and RML2016.10b are rebuilt from public mirrors of the DeepSig
  RadioML 2016 text dumps (`make_cache.py` / `make_cache_b10b.py`). The caches
  contain only the SNR subset used by the experiments. Attribution and license
  (CC BY-NC-SA, DeepSig) are noted in the manuscript; this rebuild is not the
  original distribution.
- The physical-channel second domain is generated synthetically in
  `run_b39_chansim.py` (3-tap Rayleigh + CFO + Wiener phase noise).

## Reproduce

```
python code/cvfe/tests_self.py              # unit tests + certificate checks
python code/run_b5.py --epi 2000            # synthetic grid (252 conditions)
python code/run_thm2c.py                    # multiclass bridge certificates
python code/run_complexity.py               # complexity / timing table
python code/run_b34.py --out <res>          # second-dataset deterministic half
# torch-based learning arms (GPU):
python code/run_b32.py --arms invpow_metric --n-inits 16
# split-B variant of the same protocol:
python code/run_b32.py --arms invpow_metric --n-inits 8 --split-seed 777 --split-idx 1
python code/run_b35.py --arms real,midres
python code/run_b40.py --arms real,gated,comp,midres   # second dataset, 4 arms
```

Every experiment is deterministic (seeded samplers, deterministic torch
algorithms, bit-reproducible episode streams) and writes a JSON record; the
verdict scripts recompute every number in the manuscript tables directly from
these records.

## Notes

- Server-side runners are excluded from this release; protocol parameters are
  documented in each script header and in `prereg/` (criteria fixed before
  execution, including the failure-disclosure branches).
- Known-bit reproducibility caveat: one early method column was affected by a
  dispatch bug; the fix and the paired old/new re-run are documented in the
  manuscript (Sec. canonicalization grid) and reproducible via
  `code/run_a09_canonq.py`.
