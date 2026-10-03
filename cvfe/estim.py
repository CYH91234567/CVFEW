"""原型估计器库（方向⑧ CVFEW）。

统一接口：
  proto = estimate(Zs)                     # Zs:(E,K,k,p) -> mu:(E,K,p)
  pred  = classify(Zq, proto, aux)         # Zq:(E,m,p) -> (E,m) 类别

PhaseMAP：类条件 wrapped-normal 相位先验边际化的 EM（P2(ii) Bessel 闭式 E步）。
  E步: gamma_j = E[e^{-i theta_j}|z_j] = c_{-1}/c_0
       c_n = sum_l I_l(rho) a_{n-l} e^{i l psi},  a_m = exp(-m^2 sigma_th^2 / 2)
       rho = 2|z^H mu|/sigma^2,  psi = arg(z^H mu)
  M步: mu = sum_j w_j gamma_j z_j / sum_j w_j ;  sigma^2 由期望残差更新
  极限自检: sigma_th->0 => gamma->1（欧氏均值）; sigma_th->inf, rho 大 => 硬对齐轨道。
  数值：ive(n,x)=I_n(x)e^{-x}，c_0/c_1 同用 ive（公因子 e^{rho} 相消）；
        运行时 E 步用 _gamma_grid 的平移稳定周期网格求积（rho 截断 700 以防溢出，
        大 rho 失真有界 <6e-4）；Bessel 闭式 _gamma_posterior 仅作交叉验证。
"""
import numpy as np
from scipy.special import ive

_L = 24
_RHO_MAX = 200.0


# ================================================================ 原型估计
def proto_euclid(Zs):
    return Zs.mean(axis=2)


def orbital_align(Zs, n_iter=10, restarts=3, seed=0):
    """硬轨道对齐均值（复L1-PCA目标 F(u)=sum_j|z_j^H u| 的坐标上升，多重启）。
    返回 (E,K,p)。"""
    E, K, k, p = Zs.shape
    rng = np.random.RandomState(seed)
    best_mu = np.zeros((E, K, p), dtype=Zs.dtype)
    best_obj = np.full((E, K), -np.inf)
    for r in range(restarts):
        if r == 0:
            mu = Zs.mean(axis=2)
        elif r == 1:
            mu = Zs[:, :, 0]                         # 首样本初始化
        else:
            idx = rng.randint(0, k, size=(E, K))
            mu = np.take_along_axis(Zs, idx[:, :, None, None], axis=2)[:, :, 0]
        for _ in range(n_iter):
            ip = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
            Zal = Zs * np.exp(-1j * np.angle(ip))[..., None]
            mu = Zal.mean(axis=2)
        ip = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
        obj = np.abs(ip).sum(axis=-1)
        better = obj > best_obj
        best_obj = np.where(better, obj, best_obj)
        best_mu = np.where(better[..., None], mu, best_mu)
    return best_mu


def proto_orbital(Zs):
    return orbital_align(Zs)


def proto_circcoord(Zs):
    """逐坐标纯相位 circular mean（U(1)^p 过度对齐反例，机制C）：
   丢弃每坐标幅度信息，仅同步相位 —— 合向量小的坐标由噪声主导。"""
    ph = np.angle(Zs)                                  # (E,K,k,p)
    s = np.exp(1j * ph).mean(axis=2)                   # (E,K,p) 单位幅度合成
    return s


def proto_ampweight(Zs, q_drop=0.25):
    """工程常识基线：丢低能量支持帧 + 能量归一化 + 欧氏原型。"""
    e = np.mean(np.abs(Zs) ** 2, axis=-1)              # (E,K,k)
    thr = np.quantile(e, q_drop, axis=2, keepdims=True)
    keep = e > thr
    w = keep / np.maximum(keep.sum(axis=2, keepdims=True), 1)
    Zn = Zs / np.maximum(np.sqrt(e)[..., None], 1e-12)
    return (w[..., None] * Zn).sum(axis=2)


# ================================================================ 规范化（截面）类方法
def canon_phase(Z, rule="max"):
    """规范化相位 φ(z)：使 z·e^{-iφ(z)} 落在商空间 C^p/U(1) 的一个**截面**上。

    合法性要求（旋转等变）：φ(e^{iθ}z) = φ(z) + θ ⇒ z·e^{-iφ(z)} 严格 U(1) 不变。
    四种规则都满足，区别只在**稳定性**（P7：商丛无全局连续截面 ⇒ 必有割迹）：
      max    : 取最大幅度坐标的相位（最稳的单点规则）
      wmean  : 幅度加权圆均值 Σ|z_t|²e^{i·arg z_t}（平滑规则）
      mean   : 朴素圆均值 Σz_t（低 SNR 下矢量相消）
      first  : 第一坐标实正（最差规则，割迹代价演示）
    """
    if rule == "max":
        t = np.argmax(np.abs(Z), axis=-1)
        zt = np.take_along_axis(Z, t[..., None], axis=-1)[..., 0]
        return np.angle(zt)
    if rule == "wmean":
        return np.angle((Z * np.abs(Z)).sum(-1))          # Σ|z_t|²e^{i arg z_t}
    if rule == "mean":
        return np.angle(Z.sum(-1))
    if rule == "first":
        return np.angle(Z[..., 0])
    raise ValueError(rule)


def canon_apply(Z, rule="max"):
    return Z * np.exp(-1j * canon_phase(Z, rule))[..., None]


def proto_canon(Zs, rule="max"):
    """规范化后的欧氏原型（逐帧选截面 → 均值）。"""
    return canon_apply(Zs, rule).mean(axis=2)


def cls_canon(Zq, mu, aux=None, rule="max"):
    """查询侧也规范化 → 欧氏距离（双侧截面）。"""
    Zqc = canon_apply(Zq, rule)
    d2 = (np.abs(Zqc[:, :, None, :] - mu[:, None, :, :]) ** 2).sum(-1)
    return d2.argmin(-1)


def canon_ref_vec(Zs, n_iter=6):
    """MRA 标准规范化方向：支持集（跨类池化）二阶矩的主特征向量（幂迭代）。

    Σ = Σ_j z_j z_j^H 在逐样本相位下是**相位不变**的（e^{iθ}e^{-iθ}=1），
    故其主方向 v 是固定的参考轴；v 的相位任意，但对同一 episode 内所有帧一致
    （幂迭代 coef = z^H v，v 为 Σ 的主特征向量）。
    """
    E, K, k, p = Zs.shape
    Zf = Zs.reshape(E, K * k, p)
    v = Zf.mean(1)                                        # (E,p)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    for _ in range(n_iter):
        coef = np.einsum("eip,ep->ei", Zf.conj(), v)      # (E,Kk)
        v = np.einsum("ei,eip->ep", coef, Zf)
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return v                                              # (E,p)


def canon_apply_ref(Z, v):
    """Z:(...,p) 或 (E,K,k,p)，v:(E,p)。按 φ(z)=arg(z^T v) 的相位规范化。

   为什么是 z^T v（非 z^H v）：截面须满足逐帧等变 φ(e^{iα}z)=φ(z)+α，
    z^T v 满足之；而 z^H v 是**反**等变（φ(e^{iα}z)=φ(z)−α），其代表元
    z·e^{−iφ} 会把逐帧相位 θ_j **加倍**而非剥离（数值确认：σ_θ=1.0 时
    z^H v 版准确率 43% vs z^T v 版 96.6%@σ=0.1，平坦性消失）。等价地，
    φ(z)=arg(z^T v) = −arg(z^H conj(v))，即对参考轴 conj(v) 的标准 MRA
    对齐——把逐帧全局相位**解析地**同步掉（B20 canon_ref 全轴平坦的机理）。
    """
    if Z.ndim == 3:                                       # (E,m,p)
        phi = np.angle(np.einsum("emp,ep->em", Z, v))
        return Z * np.exp(-1j * phi)[..., None]
    E, K, k, p = Z.shape
    Zf = Z.reshape(E, K * k, p)
    phi = np.angle(np.einsum("eip,ep->ei", Zf, v))
    return (Zf * np.exp(-1j * phi)[..., None]).reshape(E, K, k, p)


def proto_canon_ref(Zs):
    return canon_apply_ref(Zs, canon_ref_vec(Zs)).mean(axis=2)


def cls_canon_ref(Zq, mu, v):
    Zqc = canon_apply_ref(Zq, v)
    d2 = (np.abs(Zqc[:, :, None, :] - mu[:, None, :, :]) ** 2).sum(-1)
    return d2.argmin(-1)


# ================================================================ PhaseMAP
_PR_CACHE = {}


def _wrapped_prior(st_unique, n_grid):
    """缓存：wrapN(0, st²) 在 θ 网格上的密度。st_unique: 标量。"""
    key = (round(float(st_unique), 6), n_grid)
    if key not in _PR_CACHE:
        th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
        if key[0] < 1e-6:
            pr = np.zeros(n_grid); pr[np.argmin(np.abs(th))] = 1.0
        else:
            J = np.maximum(3, int(np.ceil(6 * key[0] / (2 * np.pi) + 3)))
            imgs = th[None, :] + 2 * np.pi * np.arange(-J, J + 1)[:, None]
            pr = np.exp(-0.5 * (imgs / key[0]) ** 2).sum(0) / (key[0] * np.sqrt(2 * np.pi))
        _PR_CACHE[key] = pr / pr.sum()
    return _PR_CACHE[key]


def _log_c0(rho, psi, st, n_grid=128):
    """log ∫ exp(ρ·cos(ψ−θ)) pr_{σ_θ}(θ) dθ —— log-sum-exp（ρ 不截断）。
    rho,psi 同形 (…)；st 标量。与 cls_marginal 内联实现同构（单测锁定等价）。"""
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    pr = _wrapped_prior(float(st), n_grid)
    LOGPR = np.where(pr > 0, np.log(np.maximum(pr, 1e-300)), -np.inf)
    m_best = np.full(rho.shape, -np.inf)
    for g0 in range(0, n_grid, 32):
        sl = slice(g0, min(g0 + 32, n_grid))
        term = rho[..., None] * (np.cos(psi[..., None] - th[sl]) - 1.0) + LOGPR[sl]
        m_best = np.maximum(m_best, term.max(-1))
    m_best = np.where(np.isfinite(m_best), m_best, 0.0)                  # 全 -inf 兜底
    acc = np.zeros(rho.shape)
    for g0 in range(0, n_grid, 32):
        sl = slice(g0, min(g0 + 32, n_grid))
        term = rho[..., None] * (np.cos(psi[..., None] - th[sl]) - 1.0) + LOGPR[sl]
        acc += np.exp(term - m_best[..., None]).sum(-1)
    return m_best + np.log(np.maximum(acc, 1e-300))


def _gamma_grid(rho, psi, sigma_th, n_grid=256):
    """E步的数值稳定路线：theta 网格求积（周期梯形=谱精度，无相消）。
    gamma = ∫ exp(rho cos(psi-θ)) pr(θ) e^{-iθ} dθ / ∫ exp(rho cos(psi-θ)) pr(θ) dθ
    稳定性：似然按 exp(rho(cos u - 1)) 平移（比率中消去，防溢出）；
    sigma_th < 1e-6 时先验为 delta(0)，解析返回 gamma=1。
    rho,psi:(E,K,k)；sigma_th:(E,K,1)。"""
    if float(np.max(sigma_th)) < 1e-6:
        return np.ones_like(rho).astype(np.complex128)
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)             # (G,)
    uth = np.unique(np.round(sigma_th, 6))                              # 缓存键
    if len(uth) == 1:
        pr = _wrapped_prior(uth[0], n_grid)[None, None, None, :]        # (1,1,1,G)
    else:
        prs = np.stack([_wrapped_prior(s, n_grid) for s in uth])        # (U,G)
        idx = np.searchsorted(uth, np.round(sigma_th, 6))
        pr = prs[idx]                                                   # (E,K,1,G)
    u = psi[..., None] - th                                             # (E,K,k,G)
    ll = np.exp(np.minimum(rho[..., None], 700.0) * (np.cos(u) - 1.0))  # 平移防溢出
    post = ll * pr                                                      # 广播 (E,K,k,G)
    num = (post * np.exp(-1j * th)).sum(-1)
    den = post.sum(-1)
    return num / np.maximum(den, 1e-300)


def _gamma_posterior(rho, psi, sigma_th):
    """E步闭式 gamma = c_{-1}/c_0 = E[e^{-i theta}|z]（P2(ii) 理论工件）。
    推导: f(θ) ∝ L(θ)pr(θ), L=Σ_l I_l e^{ilψ}e^{-ilθ}, pr=(1/2π)Σ_m a_m e^{-imθ}
          -> f ∝ Σ_n c_n e^{-inθ}, c_n = Σ_l I_l a_{n-l} e^{ilψ}
          E[e^{-iθ}] = c_{-1}/c_0 （n=-1 项）
    注意：该级数在大 rho 且 cos(psi)<0 时存在固有相位相消（c_0 = e^{rho·cos psi}
    远小于单项量级），仅用于良性参数域与网格法交叉验证；EM 运行时用 _gamma_grid。
    rho,psi:(E,K,k) 同形；sigma_th:(E,K,1) 可广播。"""
    rho_e = np.minimum(rho, _RHO_MAX)
    rho_max = float(rho_e.max()) if rho_e.size else 0.0
    L = int(min(max(_L, rho_max + 6 * np.sqrt(max(rho_max, 1.0)) + 16), 480))
    l = np.arange(-L, L + 1)                                            # (2L+1,)
    Il = ive(np.abs(l), rho_e[..., None])                               # (E,K,k,2L+1)
    eilpsi = np.exp(1j * l * psi[..., None])
    st2 = sigma_th ** 2                                                 # (E,K,1)
    a0 = np.exp(-0.5 * (l ** 2) * st2[..., None])
    a1 = np.exp(-0.5 * ((1 + l) ** 2) * st2[..., None])
    c0 = (Il * a0 * eilpsi).sum(-1)
    cm1 = (Il * a1 * eilpsi).sum(-1)
    return cm1 / np.where(np.abs(c0) > 1e-300, c0, 1.0)


def phasemap_em(Zs, sigma_th=None, kappa_meta=None, n_iter=24, tol=1e-7,
                hetero_weights=False, per_class_kappa=False, kappa_prior=None, seed=0,
                kappa_profile="joint", profile_delta=None, profile_select="argmax"):
    """PhaseMAP EM。Zs:(E,K,k,p)。

    sigma_th: 固定先验宽度（oracle/固定κ模式）；None=episode内估计（k>=2）。
    kappa_meta: 1-shot 时使用的元先验宽度（k==1 且 sigma_th=None 时生效）。
    hetero_weights: 逐样本精度加权 w_j = 1/sigma_j^2（P3 噪声异方差组件）。
    kappa_prior: (sigma0, w0) 层级收缩——episode估计按伪计数 w0 向 σ₀ 收缩
                 （经验贝叶斯；σ₀ 来自元训练/B1诊断，w0 为先验强度）。
    kappa_profile: "joint"（默认，B21）：每个 σ_θ 候选固定后重拟合 (μ,σ²)，
                   比较完整观测对数似然（真 profile likelihood）；
                   "plug"：旧行为——公共 σ² 点估计下只比较 log ĉ0 项
                   （低SNR时 σ_θ̂ 过估，保留仅作回归）。
    返回 mu:(E,K,p), aux dict（sigma2_c, sigma_th_c, n_eff_c, iters）。
    """
    E, K, k, p = Zs.shape
    mu = orbital_align(Zs, seed=seed)
    if k == 1 and sigma_th is None:
        sigma_th = kappa_meta if kappa_meta is not None else np.pi
    st = np.full((E, K), sigma_th if sigma_th is not None else 1.0)
    # 初始化 sigma^2：对齐残差
    ip = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
    Zal = Zs * np.exp(-1j * np.angle(ip))[..., None]
    sigma2 = np.maximum(np.mean(np.abs(Zal - mu[:, :, None, :]) ** 2, axis=(2, 3)),
                        1e-9)                                              # (E,K)
    w = np.ones((E, K, k))
    w_prev = None
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        ip = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)                     # (E,K,k)
        rho = 2 * np.abs(ip) / sigma2[..., None]
        psi = np.angle(ip)
        gam = _gamma_grid(rho, psi, st[..., None])                         # (E,K,k) E步（稳定网格法）
        if hetero_weights:
            sg = np.maximum((w_prev / np.max(w_prev, axis=2, keepdims=True))
                            if w_prev is not None else np.ones_like(rho), 0.0)
            # 逐样本 sigma^2 估计（残差/期望结构），向中位数收缩防过拟合
            res_j = ((np.abs(Zs) ** 2).sum(-1) + (np.abs(mu) ** 2).sum(-1)[:, :, None]
                     - 2 * np.real(gam * ip))
            s2_j = np.maximum(res_j / p, 1e-9)
            med = np.median(s2_j, axis=2, keepdims=True)
            s2_j = (8.0 * med + p * s2_j) / (8.0 + p)                      # Gamma(8, med) 经验贝叶斯收缩
            w = 1.0 / s2_j
        else:
            w = np.ones_like(rho)
        mu_new = ((w * gam)[..., None] * Zs).sum(axis=2) / w.sum(axis=2, keepdims=True)
        # sigma^2 精确 M步: E||z-e^{iθ}μ_new||^2 = ||z||^2+||mu_new||^2-2Re(gamma * z^H mu_new)
        ip_new = np.einsum("ekp,ekjp->ekj", mu_new.conj(), Zs)
        res2 = (np.abs(Zs) ** 2).sum(-1) + (np.abs(mu_new) ** 2).sum(-1)[:, :, None] \
            - 2 * np.real(gam * ip_new)
        if hetero_weights:
            sigma2_new = np.maximum((w * res2).sum(2) / np.maximum(w.sum(2), 1e-12) / p, 1e-9)
        else:
            sigma2_new = np.maximum(res2.mean(axis=2) / p, 1e-9)
        # sigma_th episode 内估计（k>=2）：跨类池化 + 有限样本去偏 + 特征函数去卷积
        # 卷积精确性: R_resid(1) = R_theta(1)·R_eps(1)（char func 相乘）；
        # R^2 去偏: E[R1^2] = 1/n + (1-1/n) R_pop^2
        if sigma_th is None and k >= 2:
            ph_res = np.angle(ip)                                          # (E,K,k)
            if per_class_kappa:
                R1 = np.abs((w * np.exp(1j * ph_res)).sum(2)) / np.maximum(w.sum(2), 1e-12)
                rho_bar = np.maximum(2.0 * np.abs(ip).mean(2) / sigma2, 1e-3)
                n_eff = np.full_like(R1, k)
            else:
                R1 = np.abs((w * np.exp(1j * ph_res)).sum((1, 2))) / np.maximum(
                    w.sum((1, 2)), 1e-12)                                  # (E,) 跨类池化
                rho_bar = (2.0 * np.abs(ip) / sigma2[..., None]).mean((1, 2))
                n_eff = np.full_like(R1, K * k)
            # 防 ive 溢出：ρ 极大时 ive(1,ρ),ive(0,ρ) 均溢出为 inf ⇒ inf/inf=NaN
            # （学习到的嵌入 σ²~1e-4 时必现）。ρ>300 时 I1/I0 → 1−1/(2ρ)，截断误差 <2e-3。
            rho_bar_c = np.minimum(np.asarray(rho_bar, dtype=float), 300.0)
            R_eps = ive(1, rho_bar_c) / np.maximum(ive(0, rho_bar_c), 1e-300)
            R_eps = np.where(np.isfinite(R_eps), R_eps, 1.0)
            R2_pop = np.maximum((n_eff * R1 ** 2 - 1.0) / np.maximum(n_eff - 1, 1.0), 1e-6)
            R_th = np.clip(np.sqrt(R2_pop) / np.maximum(R_eps, 1e-6), 1e-3, 0.9995)
            st_new = np.sqrt(np.clip(-2.0 * np.log(R_th), 1e-4, np.pi ** 2))
            if per_class_kappa:
                st = st_new
            else:
                st = np.repeat(st_new[:, None], K, axis=1)                 # episode 共享
        conv = (np.linalg.norm(mu_new - mu, axis=-1)
                / (np.linalg.norm(mu, axis=-1) + 1e-12)).max()
        mu, sigma2 = mu_new, sigma2_new
        w_prev = w
        if conv < tol:
            break
    # ---- sigma_theta 轮廓似然精化（episode 池化；对齐过拟合下比矩估计更稳）----
    if sigma_th is None and k >= 2:
        st_grid = np.array([1e-3, 0.15, 0.3, 0.5, 0.75, 1.05, 1.4, 1.8, 2.2, 2.6, 2.9, np.pi])
        if kappa_profile == "plug":
            uth = np.unique(np.round(st, 6))
            # 当前每类 sigma2 均值作为池化 sigma2
            s2_pool = sigma2.mean(axis=1)                                    # (E,)
            ip_f = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)                 # (E,K,k)
            rho_f = np.minimum(2.0 * np.abs(ip_f) / np.maximum(s2_pool[:, None, None], 1e-12), 700.0)
            psi_f = np.angle(ip_f)
            best_ll = np.full(E, -np.inf)
            best_st = np.full(E, 1.0)
            thG = np.linspace(-np.pi, np.pi, 128, endpoint=False)
            for sg in st_grid:
                pr = _wrapped_prior(sg, 128)
                ll = (rho_f[..., None] * (np.cos(psi_f[..., None] - thG) - 1.0))
                c0 = np.exp(ll).sum(-1) @ pr if False else (np.exp(ll) * pr[None, None, None, :]).sum(-1)
                # 观测数据对数似然（池化对 sigma2/μ 的公共项跨 σ_θ 不变，只差 θ 积分项）
                ll_s = np.log(np.maximum(c0, 1e-300)).sum(axis=(1, 2))       # (E,)
                better = ll_s > best_ll
                best_ll = np.where(better, ll_s, best_ll)
                best_st = np.where(better, sg, best_st)
            st = np.repeat(best_st[:, None], K, axis=1)
        else:
            # ---- B21 联合轮廓：每个候选 σ_θ 固定后重拟合 (μ,σ²)，比完整观测 LL ----
            # LL(σ_θ) = Σ[−p·log σ² − d_orb²/σ² + log ĉ0]，与 cls_marginal 同构。
            # 旧 plug 版在低SNR时被"大σ_θ→软γ→μ收缩→残差大→ĉ0虚高"的自洽环推高 σ_θ̂；
            # 每候选用自己的最优 (μ,σ²) 后该环被 ML 标准截断。
            mu0, s20 = mu, sigma2
            zs2 = (np.abs(Zs) ** 2).sum(-1)                                  # (E,K,k)
            ll_mat = np.full((E, len(st_grid)), -np.inf)
            for gi, sg in enumerate(st_grid):
                st_c = np.full((E, K), sg)
                mu_c, s2_c = mu0, s20
                for _ in range(2):                                           # 固定-σ_θ EM
                    ip_c = np.einsum("ekp,ekjp->ekj", mu_c.conj(), Zs)
                    gam_c = _gamma_grid(2.0 * np.abs(ip_c) / s2_c[..., None],
                                        np.angle(ip_c), st_c[..., None])
                    mu_c = (gam_c[..., None] * Zs).sum(2) / k
                    ip_n = np.einsum("ekp,ekjp->ekj", mu_c.conj(), Zs)
                    res2 = zs2 + (np.abs(mu_c) ** 2).sum(-1)[:, :, None] \
                        - 2 * np.real(gam_c * ip_n)
                    s2_c = np.maximum(res2.mean(2) / p, 1e-9)
                ip_f = np.einsum("ekp,ekjp->ekj", mu_c.conj(), Zs)
                aip = np.abs(ip_f)
                rho_f = 2.0 * aip / s2_c[..., None]
                d_orb2 = zs2 + (np.abs(mu_c) ** 2).sum(-1)[:, :, None] - 2.0 * aip
                ll_full = (-p * np.log(s2_c)[..., None] - d_orb2 / s2_c[..., None]
                           + _log_c0(rho_f, np.angle(ip_f), sg))
                ll_mat[:, gi] = ll_full.sum(axis=(1, 2))                     # (E,)
            best_ll = ll_mat.max(axis=1)
            am = ll_mat.argmax(axis=1)
            if profile_select == "argmax":
                best_st = st_grid[am]
            elif profile_select == "min_elig":
                # 双侧保守（B21-v2）：95% 轮廓集下界。宽相位端 LL 平坦时过度下压。
                elig = ll_mat >= (best_ll[:, None] - profile_delta)
                best_st = st_grid[elig.argmax(axis=1)]
            elif profile_select == "null_first":
                # 单侧保守（B21-v3）：σ_θ=0 为简约零假设（LRT 不拒绝则取 delta），
                # 宽相位端保持 argmax——只治"相干 episode 虚假宽度"。
                ok0 = ll_mat[:, 0] >= (best_ll - profile_delta)
                best_st = np.where(ok0, st_grid[0], st_grid[am])
            else:
                raise ValueError(profile_select)
            st = np.repeat(best_st[:, None], K, axis=1)
            for _ in range(2):                                               # 固定-κ 抛光
                ip_p = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
                gam_p = _gamma_grid(2.0 * np.abs(ip_p) / sigma2[..., None],
                                    np.angle(ip_p), st[..., None])
                mu = (gam_p[..., None] * Zs).sum(2) / k
                ip_n = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
                res2 = zs2 + (np.abs(mu) ** 2).sum(-1)[:, :, None] \
                    - 2 * np.real(gam_p * ip_n)
                sigma2 = np.maximum(res2.mean(2) / p, 1e-9)
    # ---- 层级 κ 收缩（经验贝叶斯）：伪计数 w0 向 σ₀ 收缩，再做 3 轮抛光 EM ----
    if kappa_prior is not None and sigma_th is None and k >= 2:
        s0, w0 = kappa_prior
        n_ep = float(K * k)
        st_pool = st[:, 0]                                   # 池化估计 (E,)
        st_shrunk = np.sqrt((w0 * s0 ** 2 + n_ep * st_pool ** 2) / (w0 + n_ep))
        st = np.repeat(np.maximum(st_shrunk, 1e-3)[:, None], K, axis=1)
        for _ in range(3):                                   # 固定 κ 抛光 μ, σ²
            ip_p = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
            rho_p = 2 * np.abs(ip_p) / sigma2[..., None]
            psi_p = np.angle(ip_p)
            gam_p = _gamma_grid(rho_p, psi_p, st[..., None])
            mu = (gam_p[..., None] * Zs).sum(axis=2) / k
            ip_n = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
            res2 = (np.abs(Zs) ** 2).sum(-1) + (np.abs(mu) ** 2).sum(-1)[:, :, None]                 - 2 * np.real(gam_p * ip_n)
            sigma2 = np.maximum(res2.mean(axis=2) / p, 1e-9)
    n_eff = (w * np.abs(gam) ** 2).sum(axis=2)                              # (E,K)
    aux = {"sigma2": sigma2, "sigma_th": st, "n_eff": np.maximum(n_eff, 1e-6),
           "iters": iters, "gam": gam, "w": w}
    return mu, aux


# ================================================================ 分类规则
def cls_euclid(Zq, mu, aux=None):
    d2 = ((np.abs(Zq[:, :, None, :] - mu[:, None, :, :]) ** 2)).sum(-1)
    return d2.argmin(-1)


def cls_cosine(Zq, mu, aux=None):
    num = np.einsum("emp,ekp->emk", Zq.conj(), mu).real
    den = np.linalg.norm(mu, axis=-1)[:, None, :]
    return (num / np.maximum(den, 1e-12)).argmax(-1)


def cls_hermitian(Zq, mu, aux=None):
    ip = np.einsum("emp,ekp->emk", Zq.conj(), mu)
    return ip.real.argmax(-1)


def cls_orbital(Zq, mu, aux=None, uncertainty=False):
    """轨道距离；uncertainty=True 时加组件3一阶修正（需 aux 含 sigma2,n_eff）。"""
    ip = np.einsum("emp,ekp->emk", Zq.conj(), mu)
    d2 = (np.abs(Zq) ** 2).sum(-1)[..., None] + (np.abs(mu) ** 2).sum(-1)[:, None, :] \
        - 2 * np.abs(ip)
    if uncertainty and aux is not None:
        Sig = aux["sigma2"][:, None, :] / np.maximum(aux["n_eff"][:, None, :], 1e-6)
        trS = Sig * Zq.shape[-1]
        # 一阶修正: + tr(Sig) - z^H Sig z / |z^H mu|（Sig 各向同性: z^H Sig z = Sig*||z||^2）
        d2 = d2 + trS - Sig * (np.abs(Zq) ** 2).sum(-1)[..., None] \
            / np.maximum(np.abs(ip), 1e-12)
    return d2.argmin(-1)


def cls_marginal_legacy(Zq, mu, aux, n_grid=128):
    """（旧实现，仅用于回归核对，不得进入新主张）
    原写法在低 σ²（学习到的嵌入）下会因 ρ 被 cap 到 700 而破坏二次项相消，
    明显劣化（独立补测：94.7% vs 重写版 100%）；完全退化到机会水平需更极端 σ²。"""
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    st = aux["sigma_th"]
    uth = np.unique(np.round(st, 6))
    if len(uth) == 1:
        pr = np.broadcast_to(_wrapped_prior(uth[0], n_grid)[None, None, :], (1, 1, 1) + (n_grid,))
        pr = np.broadcast_to(pr, (Zq.shape[0], 1, 1, n_grid))
    else:
        prs = np.stack([_wrapped_prior(s, n_grid) for s in uth])
        pr = prs[np.searchsorted(uth, np.round(st, 6))][:, None, :, :]
    ip = np.einsum("emp,ekp->emk", Zq.conj(), mu)
    sig2 = np.broadcast_to(aux["sigma2"].mean(axis=1)[:, None, None], ip.shape)
    rho = np.minimum(2.0 * np.abs(ip) / np.maximum(sig2, 1e-12), 700.0)
    psi = np.angle(ip)[..., None]
    c0 = np.zeros_like(ip)
    for g0 in range(0, n_grid, 32):
        sl = slice(g0, min(g0 + 32, n_grid))
        ll = np.exp(rho[..., None] * (np.cos(psi - th[sl]) - 1.0))
        c0 += (ll * pr[..., sl]).sum(-1)
    zq2 = (np.abs(Zq) ** 2).sum(-1)
    mu2 = (np.abs(mu) ** 2).sum(-1)[:, None, :]
    p_dim = Zq.shape[-1]
    score = (-p_dim * np.log(sig2) - (zq2[..., None] + mu2) / sig2 + rho
             + np.log(np.maximum(c0, 1e-300)))
    return score.argmax(-1)


def cls_marginal(Zq, mu, aux, n_grid=128):
    """生成式边际似然分类（PhaseMAP 完整模型）——数值稳定重写（2026-09-30）。

    稳定形式（与原式代数等价，但无二次项相消）：
        score_c = -p·log σ²  −  d_orb,c²/σ²  +  log ĉ0,c
        d_orb,c² = ||z||² + ||μ_c||² − 2|z^H μ_c|      （轨道距离）
        ĉ0,c     = Σ_g exp(ρ_c(cos(ψ_c − θ_g) − 1))·pr_{σ_θ,c}(θ_g),  ρ_c = 2|z^H μ_c|/σ²
        log ĉ0 用 log-sum-exp 计算（ρ 不设上限）
    极限（精确）：σ_θ→0 ⇒ log ĉ0 = ρ(cos ψ − 1) ⇒ score = −d_euclid²/σ²（欧氏）；
                  σ_θ→π ⇒ pr 均匀、ĉ0≈1 ⇒ score = −d_orb²/σ²（轨道）。
    旧实现把 ρ 截断到 700，在低 σ²（学习到的嵌入，σ²~1e-4）时使 1/σ² 量级的
    二次项失去相消、排序被 ||μ_c||² 的浮点噪声主导 ⇒ 退化为随机猜测；本实现修复之。
    """
    th = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    st = aux["sigma_th"]                                                 # (E,K)
    uth = np.unique(np.round(st, 6))
    if len(uth) == 1:
        pr1 = _wrapped_prior(uth[0], n_grid)                             # (G,)
        LOGPR = np.where(pr1 > 0, np.log(np.maximum(pr1, 1e-300)), -np.inf)
        LOGPR = LOGPR[None, None, None, :]                               # (1,1,1,G)
    else:
        prs = np.stack([_wrapped_prior(s, n_grid) for s in uth])         # (U,G)
        PR = prs[np.searchsorted(uth, np.round(st, 6))]                  # (E,K,G)
        LOGPR = np.where(PR > 0, np.log(np.maximum(PR, 1e-300)),
                         -np.inf)[:, None, :, :]                         # (E,1,K,G)
    ip = np.einsum("emp,ekp->emk", Zq.conj(), mu)                        # (E,m,K)
    sig2 = np.broadcast_to(np.maximum(aux["sigma2"].mean(axis=1), 1e-12)[:, None, None],
                           ip.shape)                                     # 池化 σ²
    aip = np.abs(ip)
    rho = 2.0 * aip / sig2                                               # 不截断
    psi = np.angle(ip)[..., None]                                        # (E,m,K,1)
    zq2 = (np.abs(Zq) ** 2).sum(-1)[..., None]                           # (E,m,1)
    mu2 = (np.abs(mu) ** 2).sum(-1)[:, None, :]                          # (E,1,K)
    d_orb2 = zq2 + mu2 - 2.0 * aip                                       # (E,m,K)
    # log ĉ0：log-sum-exp（分块控内存）
    m_best = np.full(ip.shape, -np.inf)
    for g0 in range(0, n_grid, 32):
        sl = slice(g0, min(g0 + 32, n_grid))
        term = rho[..., None] * (np.cos(psi - th[sl]) - 1.0) + LOGPR[..., sl]
        m_best = np.maximum(m_best, term.max(-1))
    m_best = np.where(np.isfinite(m_best), m_best, 0.0)                  # 全 -inf 兜底
    acc = np.zeros(ip.shape)
    for g0 in range(0, n_grid, 32):
        sl = slice(g0, min(g0 + 32, n_grid))
        term = rho[..., None] * (np.cos(psi - th[sl]) - 1.0) + LOGPR[..., sl]
        acc += np.exp(term - m_best[..., None]).sum(-1)
    logc0 = m_best + np.log(np.maximum(acc, 1e-300))
    p_dim = Zq.shape[-1]
    score = -p_dim * np.log(sig2) - d_orb2 / sig2 + logc0
    return score.argmax(-1)


def cls_mahalanobis_diag(Zq, mu, aux):
    """对角白化（距离级精度加权竞争者）。aux['var_diag']:(E,p) 查询池转导估计。"""
    v = aux["var_diag"][:, None, None, :]                    # (E,1,1,p)
    d2 = ((np.abs(Zq[:, :, None, :] - mu[:, None, :, :]) ** 2) / v).sum(-1)
    return d2.argmin(-1)


# ================================================================ 方法注册表
def whiten_aux(Zq):
    var = np.abs(Zq - Zq.mean(axis=1, keepdims=True)).mean(axis=1) ** 2 * 2   # 复方差近似
    var = np.maximum(var, np.quantile(var, 0.05, axis=1, keepdims=True))
    return {"var_diag": var}


def run_method(name, Zs, Zq, mu_true=None, kappa_meta=None, sigma_th_oracle=None):
    """单个方法在一批 episode 上端到端。返回 (pred:(E,m), aux)。"""
    if name == "euclid":
        mu = proto_euclid(Zs)
        return cls_euclid(Zq, mu), {}
    if name == "cosine":
        mu = proto_euclid(Zs)
        return cls_cosine(Zq, mu), {}
    if name == "hermitian":
        mu = proto_euclid(Zs)
        return cls_hermitian(Zq, mu), {}
    if name == "orbital":
        mu = proto_orbital(Zs)
        return cls_orbital(Zq, mu), {}
    if name == "tta_euclid":                      # 推理期TTA-min（增广在推理期的极限）
        mu = proto_euclid(Zs)
        return cls_orbital(Zq, mu), {}
    if name == "circcoord":
        mu = proto_circcoord(Zs)
        return cls_euclid(Zq, mu), {}
    if name == "whiten":
        mu = proto_euclid(Zs)
        aux = whiten_aux(Zq)
        return cls_mahalanobis_diag(Zq, mu, aux), aux
    if name == "drop_agc":
        mu = proto_ampweight(Zs)
        return cls_euclid(Zq, mu), {}
    if name in ("canon_first", "canon_mean", "canon_max", "canon_wmean"):
        rule = name.split("_", 1)[1]
        return cls_canon(Zq, proto_canon(Zs, rule), rule=rule), {}
    if name == "canonX_ref":                      # 支持侧 MRA 规范化 + 查询侧免截面
        return cls_orbital(Zq, proto_canon_ref(Zs)), {}
    if name.startswith("canonX_"):                # 支持侧选截面 + 查询侧免截面（轨道读出）
        rule = name.split("_", 1)[1]
        return cls_orbital(Zq, proto_canon(Zs, rule)), {}
    if name.startswith("canonQ_"):                # 支持侧轨道对齐 + 查询侧选截面
        rule = name.split("_", 1)[1]
        if rule == "ref":                         # 查询侧按支持集二阶矩参考轴规范化
            # A-J9 分发修复（2026-10-04）：原实现 mu = proto_canon_ref(Zs) 与
            # canon_ref 的支持侧逐字重复（分发 bug，B20/T11 已披露）。按 B18
            # 的定义，canonQ_ref 的支持侧应为**轨道原型**（估计层免截面），再与
            # 查询侧一起规范化进同一参考帧后做欧氏比较：
            v = canon_ref_vec(Zs)
            mu_orb = proto_orbital(Zs)            # 支持侧：轨道对齐（免截面估计）
            mu_c = canon_apply_ref(mu_orb, v)     # 原型进参考帧
            Zqc = canon_apply_ref(Zq, v)          # 查询侧：参考规范化（付查询侧割迹代价）
            return ((np.abs(Zqc[:, :, None, :] - mu_c[:, None, :, :]) ** 2).sum(-1)
                    .argmin(-1), {})
        return cls_canon(Zq, proto_orbital(Zs), rule=rule), {}
    if name == "canon_ref":                       # MRA 标准：二阶矩主方向规范化
        v = canon_ref_vec(Zs)
        mu = proto_canon_ref(Zs)
        return cls_canon_ref(Zq, mu, v), {}
    if name == "phasemap":
        mu, aux = phasemap_em(Zs, sigma_th=None, kappa_meta=kappa_meta)
        return cls_orbital(Zq, mu, aux, uncertainty=False), aux
    if name == "phasemap_unc":
        mu, aux = phasemap_em(Zs, sigma_th=None, kappa_meta=kappa_meta)
        return cls_orbital(Zq, mu, aux, uncertainty=True), aux
    if name == "phasemap_w":                      # + 异方差精度加权（P3组件）
        mu, aux = phasemap_em(Zs, sigma_th=None, kappa_meta=kappa_meta,
                              hetero_weights=True)
        return cls_orbital(Zq, mu, aux, uncertainty=False), aux
    if name == "phasemap_ok":                     # oracle kappa（消融上限）
        mu, aux = phasemap_em(Zs, sigma_th=sigma_th_oracle)
        return cls_orbital(Zq, mu, aux, uncertainty=False), aux
    if name == "phasemap_ml":                     # 完整模型：边际似然分类
        mu, aux = phasemap_em(Zs, sigma_th=None, kappa_meta=kappa_meta)
        return cls_marginal(Zq, mu, aux), aux
    if name == "oracle":                          # 真原型上界
        mu_b = np.broadcast_to(mu_true, Zq.shape[:1] + mu_true.shape)
        return cls_orbital(Zq, mu_b), {}
    raise ValueError(name)
