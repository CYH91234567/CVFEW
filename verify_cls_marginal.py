"""回归核对：cls_marginal 数值稳定重写 vs 旧实现。

三档对照：
  A 原始 IQ（identity 特征，σ²~O(1)）——旧实现的有效区，新旧必须**完全一致**（argmax 一致率 100%）
  B 合成 PhaseFewSyn（B5 同款）——同上
  C 学习到的嵌入（单位化、σ²~1e-4）——旧实现崩塌区，新实现必须给出合理精度
另跑 tests_self.py 全量单测。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E, synth, episodes
from cvfe.estim import cls_marginal_legacy

CACHE = r"D:\个人\CVCNN\CVXAI\04_results_from_server\radioml_cache.npz"


def acc(pred, yq):
    return float((pred == yq).mean())


def run_cell(Zs, Zq, yq, sigma_true, unit=False, tag=""):
    Zs = np.ascontiguousarray(Zs, dtype=np.complex128)
    Zq = np.ascontiguousarray(Zq, dtype=np.complex128)
    if unit:
        Zs = Zs / np.maximum(np.linalg.norm(Zs, axis=-1, keepdims=True), 1e-12)
        Zq = Zq / np.maximum(np.linalg.norm(Zq, axis=-1, keepdims=True), 1e-12)
    mu, aux = E.phasemap_em(Zs)
    p_new = E.cls_marginal(Zq, mu, aux)
    p_old = cls_marginal_legacy(Zq, mu, aux)
    mu_o, aux_o = E.phasemap_em(Zs, sigma_th=max(sigma_true, 1e-3))
    p_or = E.cls_marginal(Zq, mu_o, aux_o)
    agree = float((p_new == p_old).mean())
    print(f"  [{tag}] κ̂={np.mean(aux['sigma_th'][:,0]):.3f} σ²={np.mean(aux['sigma2']):.3e} | "
          f"new={100*acc(p_new,yq):.1f} old={100*acc(p_old,yq):.1f} agree={100*agree:.1f}% | "
          f"oracleκ={100*acc(p_or,yq):.1f} euclid={100*acc(E.cls_euclid(Zq,E.proto_euclid(Zs)),yq):.1f} "
          f"orbital={100*acc(E.cls_orbital(Zq,E.proto_orbital(Zs)),yq):.1f}")
    return agree


def main():
    print("=== 单元测试 ===")
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "cvfe", "tests_self.py")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:]); print(r.stderr[-500:])

    print("\n=== A/B：合成 PhaseFewSyn（B5 同款）===")
    for st in [0.0, np.pi / 3, np.pi]:
        mu = synth.make_prototypes(3, 64, 0.3, np.random.RandomState(0))
        Zs, Zq, yq = synth.batch_episodes(mu, 3, 5, 15, st, 0.3, 200, seed=1)
        run_cell(Zs, Zq, yq, st, unit=False, tag=f"synth σ={st:.2f}")

    if not os.path.exists(CACHE):
        print("\n[skip A] 无 RadioML 缓存"); return
    print("\n=== A：真实 IQ identity 特征（旧实现有效区）===")
    z, y, snr = D.load_radioml(CACHE)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = episodes.make_class_splits(n_splits=1)
    sp = splits[0]
    for st in [0.0, np.pi / 3, np.pi]:
        smp = episodes.EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                      q_per_class=15, seed=900, snr_min=6, snr_max=18)
        Zs, Zq, yq = smp.sample(150, inject=None if st == 0 else ("global", st))
        run_cell(Zs, Zq, yq, st, unit=False, tag=f"iq σ={st:.2f}")

    print("\n=== C：学习到的嵌入风格（单位化、支持集高度一致 ⇒ σ² 极小）===")
    rng = np.random.RandomState(3)
    for noise in [1e-2, 1e-3]:
        Kc, k, p, m = 3, 5, 32, 45
        E_ = 200
        prot = rng.randn(E_, Kc, p) + 1j * rng.randn(E_, Kc, p)
        prot /= np.abs(prot).max()
        Zs = prot[:, :, None, :] + noise * (rng.randn(E_, Kc, k, p) + 1j * rng.randn(E_, Kc, k, p))
        yq = np.repeat(np.arange(Kc), m)[None, :].repeat(E_, 0)
        Zq = prot[:, :, None, :][:, :, 0, :]  # placeholder
        Zq = np.zeros((E_, Kc * m, p), dtype=np.complex128)
        for e in range(E_):
            for c in range(Kc):
                Zq[e, c * m:(c + 1) * m] = prot[e, c] + noise * (rng.randn(m, p) + 1j * rng.randn(m, p))
        Zs = Zs / np.maximum(np.linalg.norm(Zs, axis=-1, keepdims=True), 1e-12)
        Zq = Zq / np.maximum(np.linalg.norm(Zq, axis=-1, keepdims=True), 1e-12)
        run_cell(Zs, Zq, yq, np.pi / 3, unit=False, tag=f"emb-like noise={noise:g}")


if __name__ == "__main__":
    main()
