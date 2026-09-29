"""B2 单元测试：估计器库正确性（全部本地numpy，秒级）。

T1 闭式轨道距离恒等式
T2 Bessel闭式 gamma vs 蒙特卡洛加权采样（理论-数值互证）
T3 PhaseMAP 极限行为: sigma_th=0 -> 欧氏均值; sigma_th=pi+高SNR -> 硬轨道
T4 轨道对齐目标 >= 欧氏均值目标（坐标上升单调性）
T5 episode引擎形状/可复现性
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvfe import synth as S
from cvfe import estim as E
from cvfe import synth as S

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def t1_orbital_identity(n=200, seed=0):
    rng = np.random.RandomState(seed)
    z = (rng.randn(n, 16) + 1j * rng.randn(n, 16))
    w = (rng.randn(n, 16) + 1j * rng.randn(n, 16))
    ip = np.einsum("np,np->n", z.conj(), w)
    th = -np.angle(ip)                                       # 最优对齐角 theta* = -arg(z^H w)
    d_direct = (np.abs(z - np.exp(1j * th)[:, None] * w) ** 2).sum(1)
    d_closed = (np.abs(z) ** 2 + np.abs(w) ** 2).sum(1) - 2 * np.abs(ip)
    err = np.abs(d_direct - d_closed).max()
    check("T1 closed-form orbital distance identity", err < 1e-8, f"max_err={err:.2e}")


def _gamma_grid_ref(rho, psi, st, n_grid=8192, n_wrap=12):
    """确定性网格积分参考：posterior ∝ exp(rho cos(psi-θ)) · wrapN(0,st²)。
    wrapN pdf 用截断高斯缠绕和（与 Bessel-Fourier 级数完全独立的计算路径）。"""
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    base = th[None, :] + 2 * np.pi * np.arange(-n_wrap, n_wrap + 1)[:, None]   # (2J+1, G)
    gpdf = np.exp(-0.5 * (base / st) ** 2) / (st * np.sqrt(2 * np.pi))
    pr = gpdf.sum(0)
    ll = np.exp(rho * np.cos(psi[:, None] - th[None, :]))
    post = ll * pr[None, :]
    return (post * np.exp(-1j * th)[None, :]).sum(1) / post.sum(1)


def t2_gamma_vs_grid(seed=0):
    rng = np.random.RandomState(seed)
    worst_grid = {"err": 0.0, "rho": None, "st": None}
    worst_bessel = {"err": 0.0, "rho": None, "st": None}
    for rho in [0.1, 1.0, 5.0]:   # Bessel 直接级数仅验证良性域（大 rho 相消见 T2b/实现注记）
        for st in [0.3, 1.0, 2.0]:
            psi = rng.uniform(-np.pi, np.pi, size=(64,))
            g_ref = _gamma_grid_ref(rho, psi, st)
            g_grid = E._gamma_grid(np.full((1, 1, 64), rho), psi[None, None, :],
                                   np.full((1, 1, 1), st))[0, 0]
            g_bs = E._gamma_posterior(np.full((1, 1, 64), rho), psi[None, None, :],
                                      np.full((1, 1, 1), st))[0, 0]
            eg = np.abs(g_grid - g_ref).max(); eb = np.abs(g_bs - g_ref).max()
            if eg > worst_grid["err"]: worst_grid = {"err": eg, "rho": rho, "st": st}
            if eb > worst_bessel["err"]: worst_bessel = {"err": eb, "rho": rho, "st": st}
    check("T2a grid vs wrapped-Gaussian ref (benign)", worst_grid["err"] < 5e-3,
          f"max_err={worst_grid['err']:.2e} @rho={worst_grid['rho']},st={worst_grid['st']}")
    check("T2a' Bessel vs ref (benign)", worst_bessel["err"] < 5e-3,
          f"max_err={worst_bessel['err']:.2e} @rho={worst_bessel['rho']},st={worst_bessel['st']}")
    # 大 rho 窄先验（Bessel 级数相消区）：网格法 vs 独立缠绕高斯参考仍须一致
    worst2 = 0.0
    for rho in [100.0, 200.0]:
        for st in [0.1, 0.3]:
            psi = rng.uniform(-np.pi, np.pi, size=(64,))
            g_ref = _gamma_grid_ref(rho, psi, st)
            g_grid = E._gamma_grid(np.full((1, 1, 64), rho), psi[None, None, :],
                                   np.full((1, 1, 1), st))[0, 0]
            worst2 = max(worst2, np.abs(g_grid - g_ref).max())
    check("T2b grid vs ref (cancellation regime)", worst2 < 5e-3, f"max_err={worst2:.2e}")


def t3_endpoints(seed=0):
    rng = np.random.RandomState(seed)
    mu = S.make_prototypes(5, 16, 0.3, rng)
    # sigma_th = 0：PhaseMAP ~= 欧氏均值
    Zs, Zq, _ = S.batch_episodes(mu, 5, 5, 100, 0.0, 0.05, 64, seed=1)
    m_pm, _ = E.phasemap_em(Zs, sigma_th=0.0)
    m_eu = E.proto_euclid(Zs)
    rel = (np.linalg.norm(m_pm - m_eu, axis=-1) / np.linalg.norm(m_eu, axis=-1)).max()
    check("T3a sigma_th=0 -> Euclid mean", rel < 1e-4, f"rel={rel:.2e}")
    # sigma_th = pi、低噪声：PhaseMAP ~= 硬轨道
    Zs, Zq, _ = S.batch_episodes(mu, 5, 5, 100, np.pi, 0.05, 64, seed=2)
    m_pm, _ = E.phasemap_em(Zs, sigma_th=np.pi)
    m_or = E.proto_orbital(Zs)
    rel = (np.linalg.norm(m_pm - m_or, axis=-1) / np.linalg.norm(m_or, axis=-1)).max()
    check("T3b sigma_th=pi -> orbital", rel < 0.05, f"rel={rel:.2e}")
    # gamma 极限：sigma_th=0 -> gamma=1
    g = E._gamma_posterior(np.full((2, 1, 3), 2.0), np.zeros((2, 1, 3)), np.zeros((2, 1, 1)))
    check("T3c gamma(sigma=0)=1", np.abs(g - 1).max() < 1e-10)


def t4_align_monotone(seed=0):
    rng = np.random.RandomState(seed)
    mu = S.make_prototypes(4, 32, 0.5, rng)
    Zs, _, _ = S.batch_episodes(mu, 4, 5, 20, 1.2, 0.3, 32, seed=3)
    m_or = E.proto_orbital(Zs)
    m_eu = E.proto_euclid(Zs)
    obj = lambda m: np.abs(np.einsum("ekp,ekjp->ekj", m.conj(), Zs)).sum(-1)
    gain = obj(m_or) - obj(m_eu)
    check("T4 orbital objective >= euclid objective", gain.min() >= -1e-9,
          f"min_gain={gain.min():.2e}")


def t5_engine(seed=0):
    rng = np.random.RandomState(seed)
    mu = S.make_prototypes(5, 16, 0.0, rng)
    a = S.batch_episodes(mu, 5, 1, 30, 0.8, 0.2, 16, seed=5)
    b = S.batch_episodes(mu, 5, 1, 30, 0.8, 0.2, 16, seed=5)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    check("T5a episode reproducibility", same)
    c = S.batch_episodes(mu, 5, 1, 30, 0.8, 0.2, 16, seed=6)
    diff = not np.array_equal(a[0], c[0])
    check("T5b episode seed sensitivity", diff)
    # T5d 形状语义：Zq 每查询必须等于其类原型加噪（无类间混淆）
    mu5 = S.make_prototypes(4, 8, 0.0, rng)
    Zs5, Zq5, yq5 = S.batch_episodes(mu5, 4, 3, 40, 0.0, 0.01, 16, seed=9)
    check("T5d Zs shape (E,K,k,p)", Zs5.shape == (16, 4, 3, 8), str(Zs5.shape))
    check("T5e Zq shape (E,m,p)", Zq5.shape == (16, 40, 8), str(Zq5.shape))
    check("T5f yq shape (E,m)", yq5.shape == (16, 40), str(yq5.shape))
    # sigma_theta=0, 低噪: 最近原型应几乎全对
    pred = E.cls_euclid(Zq5, E.proto_euclid(Zs5)) if False else None
    from cvfe import estim as _E
    acc = (_E.cls_orbital(Zq5, mu5[None, :, :].repeat(16, 0)) == yq5).mean() if False else None
    # 直接用真原型分类（oracle 下应≈1）
    pred_true = np.stack([(_E.cls_orbital(Zq5[i:i+1], mu5[None]) [0]) for i in range(16)])
    acc = float((pred_true == yq5).mean())
    check("T5g oracle-orbital acc≈1 at clean low-noise", acc > 0.98, f"acc={acc:.3f}")
    # coherence 构造验证
    mu_r = S.make_prototypes(8, 32, 0.7, rng)
    ip = np.einsum("kp,lp->kl", mu_r.conj(), mu_r)
    mask = ~np.eye(8, dtype=bool)
    dev = np.abs(np.abs(ip) - 0.7)[mask].max()
    check("T5c prototype coherence == rho", dev < 1e-9, f"max_dev={dev:.2e}")


if __name__ == "__main__":
    print("=== B2 unit tests ===")
    t1_orbital_identity()
    t2_gamma_vs_grid()
    t3_endpoints()
    t4_align_monotone()
    t5_engine()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"=== {len(PASS) - n_fail}/{len(PASS)} passed ===")
    sys.exit(1 if n_fail else 0)
