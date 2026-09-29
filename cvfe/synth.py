"""PhaseFewSyn：可控合成少样本任务族（方向⑧主战场）——全向量化版本。

生成模型（静态）：z_j = e^{iθ_j} h_j μ_c + ε_j,  ε_j ~ CN(0, σ_j² I_p)
  θ_j ~ wrapN(0, σ_θ²) 或 双峰 ±δ 混合（CFO模糊）
  h_j ~ 1（无衰落）或 两点混合 {1, κ_h}（比例 q 深衰落）
  σ_j ~ σ（同方差）或 两点混合 {σ, κ_σ σ}（噪声异方差）

变体：
  static    : 上述静态模型（主网格）
  cfo_ramp  : 帧内线性相位斜坡（U(1)^p 时变nuisance，检验不误伤相对相位）
  absphase  : 类条件 σ_θ,c 差异大（一半类 σ≈0）——绝对相位安全性任务(H3c)
  bimodal   : θ ~ ±δ 两点混合（CFO模糊型，H3b压力测试）

原型几何：μ_c = sqrt(1-ρ) e_c + sqrt(ρ) v（单位范数，两两内积=ρ，可控相干度）。
"""
import numpy as np


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def make_prototypes(K, p, rho, rng):
    """K 个单位范数复原型，两两相干度 |mu_c^H mu_c'| = rho（精确）。
    构造：K+1 个正交方向 q_0..q_K，mu_c = sqrt(1-rho) q_c + sqrt(rho) q_0。"""
    Q = rng.randn(K + 1, p) + 1j * rng.randn(K + 1, p)
    q, _ = np.linalg.qr(Q.T)                     # (p, K+1) 正交归一
    q = q.T                                      # (K+1, p)
    mu = np.sqrt(1 - rho) * q[1:] + np.sqrt(rho) * q[0][None, :]
    return mu.astype(np.complex128)


def batch_episodes(mu, K, k, m, sigma_theta, sigma, E, seed, variant="static",
                   fade_q=0.0, fade_kappa=0.1, noise_het_kappa=None,
                   sigma_theta_hi=None):
    """E 个 episode 的全向量化生成。Zs:(E,K,k,p) Zq:(E,m,p) yq:(E,m)"""
    p = mu.shape[1]
    rng = np.random.RandomState(seed)
    yq = rng.randint(0, K, size=(E, m))

    # ---- 支持集 (E,K,k,p) ----
    if variant == "absphase":
        st_cls = np.where(np.arange(K) < K // 2, 1e-3, (sigma_theta_hi or sigma_theta))  # (K,)
        th_s = st_cls[None, :, None] * rng.randn(E, K, k)
        th_q = st_cls[yq] * rng.randn(E, m)                 # (E,m)
    elif variant == "bimodal":
        sgn = rng.choice([-1.0, 1.0], size=(E, K, k))
        th_s = _wrap(sgn * 0.75 * np.pi + 0.05 * rng.randn(E, K, k))
        sgnq = rng.choice([-1.0, 1.0], size=(E, m))
        th_q = _wrap(sgnq * 0.75 * np.pi + 0.05 * rng.randn(E, m))
    else:
        th_s = sigma_theta * rng.randn(E, K, k)
        th_q = sigma_theta * rng.randn(E, m)

    h_s = np.ones((E, K, k))
    h_q = np.ones((E, m))
    if fade_q > 0:
        h_s[rng.rand(E, K, k) < fade_q] = fade_kappa
        h_q[rng.rand(E, m) < fade_q] = fade_kappa

    Zs = (np.exp(1j * th_s) * h_s)[..., None] * mu[None, :, None, :]   # (E,K,k,p)
    Zq = (np.exp(1j * th_q) * h_q)[..., None] * mu[yq, :]              # (E,m,p)

    if variant == "cfo_ramp":
        kk = np.arange(p) / p
        f_s = rng.randn(E, K, k) * 0.5
        f_q = rng.randn(E, m) * 0.5
        Zs = Zs * np.exp(1j * 2 * np.pi * f_s[..., None] * kk[None, None, None, :])
        Zq = Zq * np.exp(1j * 2 * np.pi * f_q[..., None] * kk[None, None, :])

    def _noise(shape):
        s = np.full(shape, sigma)
        if noise_het_kappa is not None:
            s[rng.rand(*shape) < 0.5] = sigma * noise_het_kappa
        return (rng.randn(*shape, p) + 1j * rng.randn(*shape, p)) * (s / np.sqrt(2))[..., None]

    Zs = Zs + _noise((E, K, k))
    Zq = Zq + _noise((E, m))
    return Zs, Zq, yq
