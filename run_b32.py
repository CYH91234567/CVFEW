"""B32：不变读出升级攻 SC-A 天花板（PREREG_B32，2026-10-02）。

背景（B30/B31 状态）：gated 等变臂 82.85±10.30（max 98.0、好盆 12%，init×流双盆）；
autocorr 不变臂 85.80±5.42（max 89.5、sd 腰斩但天花板低）；RealAMC 98.60。
假设 H-B32：autocorr 的天花板由 (i) 特征完备性与 (ii) 度量（等权欧氏+unit-norm）
限制，而非"不变 vs 等变"的结构限制。修复手段 = **复合不变性**（Prop B32-A）：
  Φ 逐点不变（autocorr/power）⇒ 任意 g∘Φ 逐点不变 ⇒ 学习的实线性度量头继承构造性
  不变证书（δ_inv=0 by construction，训练不可破坏；等变臂的 δ≈1e-4 只是 float 级）。

臂（ComplexAMC2(init=hann, ch=(48,96,192,48))，与 B29d/B30/B31 同 16 init×流、
3200 步、val_train 选点、σ 配对评估逐 run 配对；gated 对照不重跑，读 B30_gated.json）：
  metric        不变统计实化 → Linear(2C(τ+2)→128) → L2 → 均值原型+欧氏
  invpow        cat(autocorr, power) 不变双视图（结构+幅度）→ 均值原型+欧氏
  invpow_metric 双视图 + 度量头
  （nopool 用于补 B31 缺失的 i89 单 run：--arms nopool --out-name B30_nopool）

判据（PREREG_B32，运行前写定）：
  SC-A-v6   valsel 均值 ≥ 93 且 好盆率(≥93) ≥ 50%
  SC-Rel    部分正向：均值 ≥ 88（≥ autocorr 85.80+2pt）且 sd ≤ 6
  SC-Pair   配对 vs gated：mean_d ≥ +5pt（Wilcoxon p<0.05）
  SC-Cert   每 run δ_inv ≤ 1e-3（float32 级，实测量级 2-3e-4）且 σ 平坦 ≤ 0.1pt
            （决策级不变证书；exact 算术 δ_inv=0，float64 镜像=5e-16，tests_self T10a）
若 SC-A-v6/SC-Rel 均不过：诚实负结果（复合不变性保可靠但不提天花板），
Prop B32-A 仍成立（不变性复合是定理，与精度无关）。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, sys, time, copy
import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, invariance_error
from run_b19 import l1pca_proto, embed, heads_on, cnorm
from run_b29 import new_ema, ema_update, ema_loadable
from run_b29c import make_train_val_samplers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INIT_SEEDS = (7, 13, 17, 19, 29, 31, 41, 43, 47, 53, 59, 61, 71, 79, 83, 89)
SIG = (0.0, np.pi / 3, np.pi)
# 不变臂：嵌入逐点无全局相位 ⇒ 原型层不可对齐（l1pca 会重对齐相位差信息=B30 错误
# 在原型层重演）；均值原型+欧氏。度量头臂输出已 L2 归一（实向量），同走欧氏。
INVARIANT_ARMS = {"autocorr", "power", "invpow", "metric", "invpow_metric"}
# 输出实向量的臂（equivariance_error 对其无意义 ~|1-e^{iθ}|，改用 invariance_error）
REAL_OUT_ARMS = {"power", "metric", "invpow_metric"}


def quick_val_train(trunk, val_smps, n, arm="gated"):
    trunk.eval()
    vals = []
    for smp, st in val_smps:
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp.sample(n, inject=inj)
        es = cnorm(embed(trunk, Zs, device=DEV)).astype(np.complex128)
        eq = cnorm(embed(trunk, Zq, device=DEV)).astype(np.complex128)
        if arm in INVARIANT_ARMS:
            pred = E.cls_euclid(eq, E.proto_euclid(es))
        else:
            pred = E.cls_orbital(eq, E.proto_orbital(es))
        vals.append(float((pred == yq).mean()))
    trunk.train()
    return float(np.mean(vals))


def eval_cell_paired(trunk, zn, y, snr, sp_test, sigmas, epi_eval, seed0=7100):
    """σ 配对三头评估：同一采样器种子 ⇒ base episodes 逐位相同，注入后叠加。"""
    out = {}
    for st in sigmas:
        smp_te = EpisodeSampler(zn, y, snr, classes=sp_test, n_way=3, k_shot=5,
                                 q_per_class=15, seed=seed0, snr_min=6, snr_max=18)
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp_te.sample(epi_eval, inject=inj)
        es = embed(trunk, Zs, device=DEV).astype(np.complex128)
        eq = embed(trunk, Zq, device=DEV).astype(np.complex128)
        h, kap = heads_on(es, eq, yq, st, unit=True)
        out[f"sigma_{st:.2f}"] = {kk: float(np.mean(v)) for kk, v in h.items()}
        out[f"kappa_{st:.2f}"] = kap
        out[f"gamma_mag_{st:.2f}"] = h["gamma_mag"]
    return out


def train_one(arm, i0, zn, y, snr, sp, budget, val_every, val_n, epi_eval):
    torch.manual_seed(i0); np.random.seed(i0)
    trunk = ComplexAMC2(init="hann", pooling=arm, ch=(48, 96, 192, 48)).to(DEV)
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000, snr_min=6, snr_max=18)
    val_smps = make_train_val_samplers(zn, y, snr, sp, 88000)
    opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
    trunk.train()
    import torch.nn.functional as F
    hist = []
    ema_state = new_ema()
    best_val, best_state, best_it = -1.0, None, 0
    t0 = time.time()
    for it in range(budget):
        Zs, Zq, yq = smp_tr.sample(8)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        es = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1))
        if arm in INVARIANT_ARMS:
            mu = unit_norm(es.mean(2))                     # 均值原型（无对齐）
            d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        else:
            mu = unit_norm(l1pca_proto(es, 5))
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
            d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(ema_state, trunk)
        hist.append(float(loss.item()))
        if (it + 1) % val_every == 0:
            vt = quick_val_train(trunk, val_smps, val_n, arm=arm)
            if vt > best_val:
                best_val, best_it = vt, it + 1
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
    trunk.eval()
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
    info = {"best_val": best_val, "best_it": best_it,
            "train_min": (time.time() - t0) / 60,
            "cancel_ratio": float(cancellation_ratio(trunk, probe)),
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:])),
            "params": int(trunk.n_params())}
    if arm in REAL_OUT_ARMS:
        info["invariance_delta"] = float(invariance_error(trunk, probe, device=DEV))
    else:
        info["equiv_delta"] = float(equivariance_error(trunk, probe, device=DEV))
    sets = {"endpoint": {kk: vv.detach().cpu().clone() for kk, vv in trunk.state_dict().items()}}
    if best_state is not None:
        sets["valsel"] = best_state
    if ema_state["state"] is not None:
        sets["ema"] = ema_loadable(ema_state, trunk)
    ev = {}
    for name, sd in sets.items():
        twin = copy.deepcopy(trunk)
        twin.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
        twin.eval()
        ev[name] = eval_cell_paired(twin, zn, y, snr, sp["test"], SIG, epi_eval)
        head = "euclid" if arm in INVARIANT_ARMS else "orbital"
        accs = {s: 100 * ev[name][f"sigma_{s:.2f}"][head] for s in SIG}
        print(f"  [{arm}|i{i0}|{name}] " +
              " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
              f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
        del twin
    return info, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--out-name", default=None, help="落盘 json 名（默认 B32_{arm}.json）")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--arms", default="metric,invpow,invpow_metric")
    ap.add_argument("--budget", type=int, default=3200)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]

    arms = a.arms.split(",")
    if a.mode == "smoke":
        arms, budget, val_every, val_n, epi_eval = tuple(a.arms.split(",")), 60, 20, 10, 20
    else:
        budget, val_every, val_n, epi_eval = a.budget, 100, 100, a.epi_eval

    for arm in arms:
        oname = a.out_name or f"B32_{arm}.json"
        if not oname.endswith(".json"):            # --out-name 须含 .json，否则静默写旁路文件
            oname += ".json"
        res_path = os.path.join(a.out, "logs", oname)
        recs = {"meta": {"prereg": "PREREG_B32.md (B32-A 复合不变性)", "arm": arm,
                         "budget": budget, "inits": list(INIT_SEEDS), "stream_seed": 4000,
                         "eval": "sigma-paired (same base episodes across sigma)",
                         "disjoint": True, "deterministic": True,
                         "protocol": "B29c/B29d/B30/B31 identical"}, "runs": {}}
        if os.path.exists(res_path):
            old = json.load(open(res_path))
            recs["runs"].update(old.get("runs", {}))
            recs["meta"] = old.get("meta", recs["meta"])
            print(f"[resume] {res_path}: {len(recs['runs'])} done", flush=True)
        t00 = time.time()
        for i0 in INIT_SEEDS:
            key = f"{arm}_i{i0}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            print(f"[run] {key}", flush=True)
            info, ev = train_one(arm, i0, zn, y, snr, sp, budget, val_every,
                                 val_n, epi_eval)
            recs["runs"][key] = {"info": info, "eval": ev}
            json.dump(recs, open(res_path, "w"), indent=1)
        print(f"B32[{arm}] DONE {(time.time()-t00)/60:.1f}min -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
