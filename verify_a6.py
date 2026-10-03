"""A.6 Mixture-futility 引理的数值证书（三种数据情形，诚实区分不变性层级）。

情形 A（良态结构化数据）：类有强公共方向 + 小噪声（RadioML 嵌入式协议的代理）。
情形 B（退化数据）：各向同性高斯（同步目标退化、多盆——对应 B4/P2iii）。
情形 C（整集公共相位）：所有帧共享一个 φ（原 sketch 证明的成立域）。

验证的命题层级：
  (i) 目标级：J(μ)=Σ_j|μ^H e_j| 逐项逐帧旋转不变（精确）；
  (ii) 算法级：坐标上升更新映射不变；**不变初始化**（首样本/公共相位均值）
      ⇒ 原型逐位相同；**相干均值初始化**（run_b19.l1pca_proto /
      estim.orbital_align 的实际默认）在良态数据上恢复全局最优⇒逐位相同，
      在退化数据上可能落入不同盆⇒损失不同（目标值仍不变）；
  (iii) 模长读出：d² 与 CE 损失在原型相同时逐点相同；
  (iv) 不变统计读出（|h|、帧内自积 h(t+τ)conj(h(t))）逐帧相位逐点不变；
  (v) 负对照：相干均值+欧氏头改变；帧内 Wiener pnoise 改变。

输出：04_results/logs/A06_mixture_futility_cert.json
"""
import json, os
import numpy as np

rng = np.random.RandomState(20261003)
P, L, K, Q = 12, 64, 5, 30
NIT = 5


def make_structured(rng, n_cls=2):
    """类有强公共方向（良态同步问题：唯一全局最优）。"""
    mus = rng.randn(n_cls, L) + 1j * rng.randn(n_cls, L)
    mus = mus / np.linalg.norm(mus, axis=1, keepdims=True)
    Zs, ys = [], []
    for c in range(n_cls):
        for j in range(K // n_cls + (1 if c == 0 else 0)):
            sig = 0.25
            Zs.append(mus[c] * 1.0 + sig * (rng.randn(L) + 1j * rng.randn(L)))
            ys.append(c)
    ys = np.array(ys)
    Zq = np.stack([mus[c] * 1.0 + 0.25 * (rng.randn(L) + 1j * rng.randn(L))
                   for c in rng.randint(0, n_cls, Q)])
    yq = rng.randint(0, n_cls, Q)
    return np.stack(Zs), ys, Zq, yq


def make_degenerate(rng):
    """各向同性高斯（同步目标退化、多盆）。"""
    return (rng.randn(K, L) + 1j * rng.randn(K, L), np.arange(K) % 2,
            rng.randn(Q, L) + 1j * rng.randn(Q, L), rng.randint(0, 2, Q))


def l1pca(E_, n_it=NIT, init="mean"):
    """坐标上升同步原型；init: mean（实现默认）/ first（不变初始化）。"""
    mu = E_.mean(0) if init == "mean" else E_[0].copy()
    mu = mu / np.linalg.norm(mu)
    for _ in range(n_it):
        w = np.exp(-1j * np.angle(np.conj(mu) @ E_.T))
        mu = (w[:, None] * E_).sum(0)
        n = np.linalg.norm(mu)
        mu = mu / n if n > 1e-12 else mu
    return mu


def ce_loss(Es, Eq, ys, yq, proto_fn):
    n_cls = ys.max() + 1
    mus = np.stack([proto_fn(Es[ys == c]) for c in range(n_cls)])
    d2 = (np.sum(np.abs(Eq) ** 2, 1)[:, None] + np.sum(np.abs(mus) ** 2, 1)[None, :]
          - 2 * np.abs(Eq @ np.conj(mus).T))
    logits = -d2
    lse = logits.max(1, keepdims=True)
    ce = (~(yq[:, None] == np.arange(n_cls))) * (logits - lse - np.log(np.exp(logits - lse).sum(1, keepdims=True)))
    return float(np.abs(ce).sum())


def obj_value(Es, ys, mu_dirs):
    """J 在**固定随机方向**上的取值——目标级（i）的直接证书（与 μ̂ 无关）。"""
    n_cls = ys.max() + 1
    tot = 0.0
    for c in range(n_cls):
        E_ = Es[ys == c]
        mu = mu_dirs[c]
        tot += float(np.abs(np.conj(mu) @ E_.T).sum())
    return tot


def inv_stats(E_):
    m2 = np.sum(np.abs(E_) ** 2, 1)
    R = E_[:, 1:] * np.conj(E_[:, :-1])
    return np.concatenate([m2, np.sum(np.abs(R) ** 2, 1)])


def rotate_perframe(Z, r):
    return Z * np.exp(1j * r.randn(Z.shape[0], 1))


def rotate_common(Z, r):
    return Z * np.exp(1j * r.randn())


def pnoise(Z, r, strength=0.5):
    return Z * np.exp(1j * np.cumsum(strength * r.randn(*Z.shape), axis=1))


out = {"regimes": {}}
for name, (Zs, ys, Zq, yq) in (("structured", make_structured(rng)),
                               ("degenerate", make_degenerate(rng))):
    W = rng.randn(P, L) + 1j * rng.randn(P, L)          # 任意复线性 trunk（严格等变）
    Es0, Eq0 = Zs @ W.T, Zq @ W.T
    Es1 = rotate_perframe(Es0, rng)                      # 逐帧独立相位
    Ec = rotate_common(Es0, rng)                         # 整集公共相位
    n_cls = ys.max() + 1
    mu_dirs = rng.randn(n_cls, P) + 1j * rng.randn(n_cls, P)
    mu_dirs = mu_dirs / np.linalg.norm(mu_dirs, axis=1, keepdims=True)
    r = {"obj_orig": obj_value(Es0, ys, mu_dirs), "obj_perframe": obj_value(Es1, ys, mu_dirs),
         "obj_common": obj_value(Ec, ys, mu_dirs)}
    for init in ("mean", "first"):
        loss = lambda E: ce_loss(E, Eq0, ys, yq, lambda X: l1pca(X, init=init))
        lf = ce_loss(Es1, Eq0, ys, yq, lambda X: l1pca(X, init=init)) if False else None
        r[f"loss_perframe_{init}"] = {
            "orig": loss(Es0), "perframe": ce_loss(Es1, Eq0, ys, yq, lambda X: l1pca(X, init=init)),
            "common": ce_loss(Ec, Eq0, ys, yq, lambda X: l1pca(X, init=init))}
        for k in ("perframe", "common"):
            r[f"loss_perframe_{init}"][f"max_abs_diff_{k}"] = abs(
                r[f"loss_perframe_{init}"][k] - r[f"loss_perframe_{init}"]["orig"])
    r["inv_stats_perframe_diff"] = float(np.abs(inv_stats(Es1) - inv_stats(Es0)).max())
    r["euclid_mean_perframe_diff"] = abs(
        ce_loss(Es0, Eq0, ys, yq, lambda X: X.mean(0))
        - ce_loss(Es1, Eq0, ys, yq, lambda X: X.mean(0)))
    # pnoise 负对照（帧内 Wiener）
    Ep = pnoise(Zs, rng) @ W.T
    r["pnoise_diff"] = abs(ce_loss(Es0, Eq0, ys, yq, lambda X: l1pca(X))
                           - ce_loss(Ep, Eq0, ys, yq, lambda X: l1pca(X)))
    out["regimes"][name] = r

v = {}
for regime in ("structured", "degenerate"):
    r = out["regimes"][regime]
    v[f"{regime}_obj_invariant"] = abs(r["obj_perframe"] - r["obj_orig"]) < 1e-10
    v[f"{regime}_mean_init_noop"] = r["loss_perframe_mean"]["max_abs_diff_perframe"] < 1e-10
    v[f"{regime}_first_init_noop"] = r["loss_perframe_first"]["max_abs_diff_perframe"] < 1e-10
    v[f"{regime}_common_phase_noop"] = r["loss_perframe_mean"]["max_abs_diff_common"] < 1e-10
    v[f"{regime}_inv_stats_noop"] = r["inv_stats_perframe_diff"] < 1e-10
out["verdict"] = v
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_results", "logs",
                 "A06_mixture_futility_cert.json")
json.dump(out, open(p, "w"), indent=1)
print(json.dumps(v, indent=1))
for regime, r in out["regimes"].items():
    print(f"[{regime}] obj {r['obj_orig']:.6f}/{r['obj_perframe']:.6f}; "
          f"loss mean-init diff {r['loss_perframe_mean']['max_abs_diff_perframe']:.3e}; "
          f"first-init diff {r['loss_perframe_first']['max_abs_diff_perframe']:.3e}; "
          f"common-phase diff {r['loss_perframe_mean']['max_abs_diff_common']:.3e}; "
          f"inv-stats {r['inv_stats_perframe_diff']:.3e}; "
          f"euclid {r['euclid_mean_perframe_diff']:.3e}; pnoise {r['pnoise_diff']:.3e}")
print("A06 cert ->", p)
