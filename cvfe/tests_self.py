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


def t6_log_c0_equivalence(n=64, seed=0):
    """T6 (B21) _log_c0 与 cls_marginal 内联 log-sum-exp 实现数值等价；
    且 σ_θ→0 极限 = ρ(cos ψ−1)（delta 先验）。"""
    rng = np.random.RandomState(seed)
    rho = rng.uniform(0.1, 40.0, (n,))
    psi = rng.uniform(-np.pi, np.pi, (n,))
    for sg in (0.3, 1.05, np.pi):
        th = np.linspace(-np.pi, np.pi, 128, endpoint=False)
        pr = E._wrapped_prior(sg, 128)
        ll = (rho[..., None] * (np.cos(psi[..., None] - th) - 1.0))
        c0 = (np.exp(ll - ll.max(-1, keepdims=True)) * pr[None, :]).sum(-1)
        ref = ll.max(-1) + np.log(np.maximum(c0, 1e-300))
        got = E._log_c0(rho, psi, sg)
        err = np.abs(got - ref).max()
        check(f"T6a _log_c0 == inline lse (st={sg:.2f})", err < 1e-8, f"max_err={err:.2e}")
    got0 = E._log_c0(rho, psi, 1e-3)
    err0 = np.abs(got0 - rho * (np.cos(psi) - 1.0)).max()
    check("T6b _log_c0 delta-prior limit", err0 < 1e-6, f"max_err={err0:.2e}")


def t7_joint_profile_endpoints(n=48, seed=0):
    """T7 (B21) 联合轮廓 σ_θ̂：σ_θ=0 → 小；σ_θ=π/3 → 中段。k=5, p=16, ρ=0.3。"""
    from cvfe import synth as _S
    rng = np.random.RandomState(seed)
    mu = _S.make_prototypes(5, 16, 0.3, rng)
    for st_true, lo, hi in ((0.0, 0.0, 0.15), (np.pi / 3, 0.4, 1.5)):
        Zs, _, _ = _S.batch_episodes(mu, 5, 5, 4, st_true, 0.5, n, seed=31 + int(st_true * 7))
        _, aux = E.phasemap_em(Zs, kappa_profile="joint")
        med = float(np.median(aux["sigma_th"][:, 0]))
        check(f"T7 joint profile st_hat at σ_θ={st_true:.2f} in [{lo},{hi}]",
              lo <= med <= hi, f"median={med:.3f}")


def t8_canon_ref_section(n=512, seed=0):
    """T8（审计轮）锁定 canon_ref 截面性质，防止把代码"修"成文档误写的 z^H v。
    T8a 逐帧相位剥离：固定 v 下代表元 z·e^{-iφ(z)} 对帧旋转不变（3-D/4-D 两路径）。
    T8b 逐帧等变：φ(e^{iα}z)=φ(z)+α（z^T v）；z^H v 反等变（加倍相位）——
       部署版在 σ_θ=1.0 必须显著优于 z^H v 误版。"""
    rng = np.random.RandomState(seed)
    mu = S.make_prototypes(5, 16, 0.3, rng)
    Zs, Zq, yq = S.batch_episodes(mu, 5, 5, 75, 1.0, 0.2, n, seed=23)
    v = E.canon_ref_vec(Zs)
    # T8a：固定 v 的代表元对帧旋转不变（逐帧相位剥离）
    for path, Z in (("3D", Zq[:8]), ("4D", Zs[:8])):
        rep0 = E.canon_apply_ref(Z, v[:8] if path == "3D" else v[:8])
        rot = np.exp(1j * 0.37)
        rep1 = E.canon_apply_ref(Z * rot, v[:8])
        err = np.abs(rep0 - rep1).max()
        check(f"T8a rep invariant to frame rotation ({path} path)", err < 1e-10,
              f"max_err={err:.2e}")
    # T8b：σ_θ=1.0 下部署版 vs z^H v 误版
    def classify_with(fn):
        v_ = E.canon_ref_vec(Zs)
        muc = fn(Zs, v_).mean(axis=2)
        Zqc = fn(Zq, v_)
        d2 = (np.abs(Zqc[:, :, None, :] - muc[:, None, :, :]) ** 2).sum(-1)
        return float((d2.argmin(-1) == yq).mean())
    def apply_H(Z, v_):
        if Z.ndim == 3:
            phi = np.angle(np.einsum("emp,ep->em", np.conj(Z), v_))
            return Z * np.exp(-1j * phi)[..., None]
        E_, K, k, p = Z.shape
        Zf = Z.reshape(E_, K * k, p)
        phi = np.angle(np.einsum("eip,ep->ei", np.conj(Zf), v_))
        return (Zf * np.exp(-1j * phi)[..., None]).reshape(E_, K, k, p)
    a_T = classify_with(E.canon_apply_ref)
    a_H = classify_with(apply_H)
    check("T8b TT section >= HH misread +20pt @ sigma_th=1.0", a_T >= a_H + 0.20,
          f"TT={a_T:.3f} HH={a_H:.3f}")


def _orbit_pool_np(h, n_iter=5, lam=1.0):
    """orbit_pool 的 numpy 镜像（与 nets_b19.orbit_pool 结构逐字同构；
    torch 版本机不可用，其等变性在服务器由 equivariance_error 实测）。"""
    u = h.mean(-1)
    for _ in range(n_iter):
        ip = np.conj(u[..., None]) * h
        w = np.exp(-1j * lam * np.angle(ip))
        u = (h * w).mean(-1)
    return u


def t9_orbit_pool_equivariance(seed=0):
    """T9（B30）：轨道池化（时间轴 L1-PCA）的三个性质（numpy 镜像，与实现无关）。
    T9a 池化层等变：h→e^{iθ}h ⇒ u→e^{iθ}u。
    T9b 相干增益：隐层相位近均匀（坏盆机制）时 orbit 嵌入幅度 >> mean 嵌入。
    T9c L1-PCA 目标单调：obj(orbit) >= obj(mean)。"""
    rng = np.random.RandomState(seed)
    # T9a
    h = rng.randn(8, 32, 64) + 1j * rng.randn(8, 32, 64)
    u0 = _orbit_pool_np(h)
    worst = 0.0
    for th in (0.7, np.pi / 2, np.pi, 2.2):
        ur = _orbit_pool_np(h * np.exp(1j * th))
        worst = max(worst, float(np.abs(ur - u0 * np.exp(1j * th)).max()
                                 / np.abs(u0).max()))
    check("T9a orbit-pool equivariance", worst < 1e-12, f"rel_err={worst:.2e}")
    # T9b：相位完全打散的隐层
    ph = rng.uniform(-np.pi, np.pi, h.shape)
    h_rand = h * np.exp(1j * ph)
    em, eo = h_rand.mean(-1), _orbit_pool_np(h_rand)
    gain = float(np.abs(eo).max(axis=-1).mean() / max(np.abs(em).max(axis=-1).mean(), 1e-12))
    check("T9b orbit-pool coherence gain over mean (phase-scattered hidden)", gain > 3.0,
          f"gain={gain:.2f}x（理论量级 ~√L 折扣后的同步收益）")
    # T9c
    def obj(e):
        return np.abs(np.conj(e[..., None]) * h_rand).sum(-1)
    om, oo = obj(h_rand.mean(-1)), obj(_orbit_pool_np(h_rand))
    check("T9c orbit objective >= mean objective", float((oo - om).min()) >= -1e-8,
          f"min_gain={float((oo - om).min()):.2e}")


def _invariant_stats_np(h, tau_max=6):
    """invariant_stats 的 numpy 镜像（性质与实现无关）。"""
    L = h.shape[-1]
    feats = [(h[..., tau:] * np.conj(h[..., : L - tau])).mean(-1)
             for tau in range(1, tau_max + 1)]
    a2 = np.abs(h) ** 2
    feats += [a2.mean(-1), (a2 ** 2).mean(-1)]
    return np.stack(feats, axis=-1)


def t10_invariant_stats(seed=0):
    """T10（B31）：自相关/矩统计量的**精确不变性**与结构保持。
    T10a 不变性：h→e^{iθ}h ⇒ e 完全不变（非仅等变）。
    T10b 结构保持：相位差信息保留（两信号仅时间相位轨迹不同 ⇒ 特征不同）。"""
    rng = np.random.RandomState(seed)
    h = rng.randn(8, 32, 64) + 1j * rng.randn(8, 32, 64)
    e0 = _invariant_stats_np(h)
    worst = 0.0
    for th in (0.7, np.pi / 2, np.pi, 2.2):
        e1 = _invariant_stats_np(h * np.exp(1j * th))
        worst = max(worst, float(np.abs(e1 - e0).max() / max(np.abs(e0).max(), 1e-12)))
    check("T10a invariant_stats exact invariance", worst < 1e-14, f"rel={worst:.2e}")
    # T10b：时间相位轨迹不同的两信号 ⇒ 自相关相位不同（结构被保留）
    ph = 0.3 * np.arange(64)                                  # 线性相位斜坡（CFO 型）
    h2 = h * np.exp(1j * ph)[None, None, :]
    d = np.abs(_invariant_stats_np(h) - _invariant_stats_np(h2)).max()
    base = np.abs(_invariant_stats_np(h)).max()
    check("T10b autocorr preserves phase-structure (CFO ramp changes features)",
          d > 0.05 * base, f"rel_change={d/base:.3f}")
    # T10c：幅度缩放按齐次性缩放统计量（R(τ)/m2 二次、m4 四次），相位不受污染
    hs = h * 2.0
    e_s = _invariant_stats_np(hs)
    scale = np.ones_like(e0)          # 前 tau_max+1 项二次、最后 m4 四次
    scale[..., -1] = 16.0
    scale[..., :-1] = 4.0
    check("T10c amplitude scaling follows degree homogeneity",
          float(np.abs(e_s - scale * e0).max() / np.abs(e0).max()) < 1e-12)


if __name__ == "__main__":
    print("=== B2 unit tests ===")
    t1_orbital_identity()
    t2_gamma_vs_grid()
    t3_endpoints()
    t4_align_monotone()
    t5_engine()
    t6_log_c0_equivalence()
    t7_joint_profile_endpoints()
    t8_canon_ref_section()
    t9_orbit_pool_equivariance()
    t10_invariant_stats()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"=== {len(PASS) - n_fail}/{len(PASS)} passed ===")
    sys.exit(1 if n_fail else 0)
