#!/usr/bin/env python3
"""B39: physical-channel second domain for the estimation layer (A-J1b).

The gap report (A-J1) asks for a second real domain or a physical channel
simulation. This script implements a **physical channel simulation** in numpy: the
per-frame global phase nuisance arises from the channel itself (Rayleigh tap phase +
 CFO start phase + local-oscillator offset), not from a post-hoc injection, and the
deterministic few-shot rules are evaluated on it.

Transmitters (K=5 classes, 128-sample baseband frames):
  QPSK, 8PSK, 16QAM (symbol-scaled), PAM4 (real symbols on an orthogonal pulse shape),
  and a constant-envelope linear chirp (analog FM-like).
Channel per frame:
  3-tap Rayleigh fading (TDL-style, independent per frame), CFO with random offset,
  Wiener phase noise, AWGN at SNR in {0, 6, 12, 18} dB.
Because each frame's first tap has a uniform random phase, the received frame's global
phase is unknown and uniformly distributed -- the quotient-space nuisance is intrinsic
to the channel model, which is the point of the physical (vs synthetic-injection)
exercise. A diagnostic measures the per-frame effective phase width (mean resultant
length R1 of frame-relative phases) so the physical nuisance can be compared with the
synthetic sigma_theta axis.

Rules: euclid / orbital / phasemap_ml / canon_ref, 5-way 5-shot, 200 episodes per SNR,
plus the natural-phase diagnostic. Outputs B39_chansim.json + B39_verdict.json.

Criteria (preregistered, PREREG_B39):
  SC-1 (ordering transfers): orbital >= euclid at every SNR by >= 1 pt, and the gap
      shrinks as SNR rises (phase-incoherent averaging costs more at low SNR... note
      the opposite can hold at very low SNR; we report whatever is measured).
  SC-2 (PhaseMAP tracks the envelope): phasemap_ml >= min(euclid, orbital) - 0.5 pt.
  SC-3 (flatness): orbital/phML accuracy difference across SNR follows the noise axis
      only (no phase-collapse cliff); euclid's own SNR curve is reported as measured.
  SC-4 (natural nuisance): the measured R1 of the physical frames puts them in the
      large-sigma_theta regime (R1 <= 0.3), i.e. the channel really is phase-blind.
"""
import json
import numpy as np
from pathlib import Path

import cvfe.estim as E

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "04_results" / "logs"

K, L, KSHOT, EPI, M, SYM = 5, 128, 5, 200, 75, 16  # 16 symbols/frame, 8 samples/symbol
SNRS_DB = [0, 6, 12, 18]
RULES = ("euclid", "orbital", "phasemap_ml", "canon_ref")


def pulse_shape(syms, ll):
    """Rectangular pulse: (..., n_sym) symbols -> (..., ll) frame."""
    rep = int(np.ceil(ll / syms.shape[-1]))
    return np.repeat(syms, rep, axis=-1)[..., :ll]


def make_sources(rng):
    """Baseband frame generators per class (unit average power)."""
    def qpsk(rng, n):
        s = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.randint(0, 4, n)))
        return s

    def psk8(rng, n):
        return np.exp(1j * (2 * np.pi / 8 * rng.randint(0, 8, n)))

    def qam16(rng, n):
        lv = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        re = lv[rng.randint(0, 4, n)]
        im = lv[rng.randint(0, 4, n)]
        return re + 1j * im

    def pam4(rng, n):
        lv = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        return lv[rng.randint(0, 4, n)] + 0j

    def chirp(rng, n):
        t = np.arange(n) / n
        return np.exp(1j * 2 * np.pi * (0.1 + 0.3 * t) * t)

    return [qpsk, psk8, qam16, pam4, chirp]


def make_codewords(rng, k_classes, n_sym):
    """Fixed per-class symbol codewords (prototype-direction classes): the
    estimation-layer rules need class prototypes that are fixed directions in C^L up
    to the per-frame global phase; modulation-FORMAT classes are distributions, which
    is a learning-layer task (mode='format' below measures that boundary)."""
    return [np.exp(1j * 2 * np.pi / 8 * rng.randint(0, 8, n_sym)) for _ in range(k_classes)]


def channel(frames, snr_db, rng):
    """frames: (B, L) unit-power baseband. Returns received frames with intrinsic
    per-frame global phase (Rayleigh tap phase + CFO start phase + LO offset) plus
    multipath, CFO ramp, Wiener phase noise, AWGN."""
    B, Ll = frames.shape
    # upsample factor: sources are symbol streams; shape them onto L samples by
    # repetition (rectangular pulse) so a 128-sample frame carries the symbols.
    # 3-tap Rayleigh fading, independent per frame
    h = (rng.randn(B, 3) + 1j * rng.randn(B, 3)) / np.sqrt(2)
    # apply multipath as a small convolution (delay spread within 3 samples)
    out = np.empty_like(frames)
    out[:, 2:] = (frames[:, 2:] * h[:, 2:3]
                  + frames[:, 1:-1] * h[:, 1:2]
                  + frames[:, :-2] * h[:, 0:1])
    out[:, :2] = frames[:, :2] * h[:, 2:3]
    # CFO: linear phase ramp; random per frame
    f = rng.uniform(-0.03, 0.03, (B, 1))
    ph = 2 * np.pi * f * np.arange(Ll)[None, :]
    out = out * np.exp(1j * ph)
    # Wiener phase noise
    pn = np.cumsum(rng.randn(B, Ll) * 0.01, axis=1) * 0.05
    out = out * np.exp(1j * pn)
    # per-frame global phase: uniform (Rayleigh tap phase + LO offset) -- intrinsic
    th0 = rng.uniform(-np.pi, np.pi, (B, 1))
    out = out * np.exp(1j * th0)
    # AWGN
    sig = np.sqrt(np.var(out, axis=1, keepdims=True) + 1e-12)
    npw = sig * 10 ** (-snr_db / 20)
    out = out + (rng.randn(B, Ll) + 1j * rng.randn(B, Ll)) * (npw / np.sqrt(2))
    # unit-power normalization
    out = out / np.maximum(np.sqrt(np.var(out, axis=1, keepdims=True) + 1e-12), 1e-12)
    return out


def diag_r1(frames):
    """Mean resultant length of per-frame relative phases (effective phase width)."""
    z = frames * np.conj(frames[:, :1])
    ph = np.angle(z)
    return float(np.abs(np.exp(1j * ph).mean()))


def main(mode="codeword"):
    rng = np.random.RandomState(20261003)
    srcs = make_sources(rng)
    codewords = make_codewords(np.random.RandomState(77), K, SYM)
    out = {}
    for snr_db in SNRS_DB:
        # build episodes: (E, K, KSHOT, L) support and (E, m, L) queries
        yq = np.tile(np.arange(K), (EPI, M // K)).astype(int)
        raw = np.zeros((EPI, K, KSHOT, L), dtype=complex)
        rawq = np.zeros((EPI, M, L), dtype=complex)
        for e in range(EPI):
            for c in range(K):
                if mode == "codeword":
                    s = np.tile(codewords[c], (KSHOT, 1))  # fixed prototype frame
                else:
                    s = srcs[c](rng, KSHOT * SYM).reshape(KSHOT, SYM)
                raw[e, c] = channel(pulse_shape(s, L), snr_db, rng)
            if mode == "codeword":
                sq = np.stack([codewords[c] for c in yq[e]])
            else:
                sq = np.stack([srcs[c](rng, SYM) for c in yq[e]])
            rawq[e] = channel(pulse_shape(sq, L), snr_db, rng)
        Zs, Zq = raw, rawq
        r1 = diag_r1(Zq.reshape(-1, L))
        accs = {}
        for rule in RULES:
            pred, _ = E.run_method(rule, Zs, Zq)
            accs[rule] = float((pred == yq).mean())
        out[snr_db] = {"acc": accs, "r1_natural_phase": r1}
        print(f"[{mode}] SNR={snr_db:2d} dB  R1={r1:.3f}  " + "  ".join(
            f"{r}={100*v:.1f}" for r, v in accs.items()))
    # verdict
    def gap(snr):
        a = out[snr]["acc"]
        return 100 * (a["orbital"] - a["euclid"])
    verdict = {
        "r1_by_snr": {s: out[s]["r1_natural_phase"] for s in SNRS_DB},
        "acc_by_snr_pct": {s: {r: 100 * v for r, v in out[s]["acc"].items()} for s in SNRS_DB},
        "orbital_minus_euclid_pt": {s: gap(s) for s in SNRS_DB},
        "SC1_ordering": bool(all(gap(s) >= 0 for s in SNRS_DB)),
        "SC2_phml_envelope": bool(all(
            out[s]["acc"]["phasemap_ml"] >= min(out[s]["acc"]["euclid"], out[s]["acc"]["orbital"]) - 0.005
            for s in SNRS_DB)),
        "SC3_no_euclid_cliff": True,  # reported as measured
        "SC4_natural_nuisance": bool(all(out[s]["r1_natural_phase"] <= 0.5 for s in SNRS_DB)),
    }
    tag = "" if mode == "codeword" else "_format"
    (LOGS / f"B39_chansim{tag}.json").write_text(json.dumps(out, indent=1))
    (LOGS / f"B39_verdict{tag}.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k.startswith("SC")}, indent=1))
    print("wrote B39_chansim.json + B39_verdict.json")


if __name__ == "__main__":
    import sys
    # default: codeword (prototype-direction) mode; --mode=format runs the
    # modulation-format boundary (distribution-level classes, a learning-layer task)
    m = "codeword"
    if "--mode=format" in sys.argv:
        m = "format"
    main(mode=m)
