"""B30：轨道池化（分析式相位同步）攻 SC-A 双盆（PREREG_B30）。

3 臂（gated 对照 / orbit 主臂 / hybrid 保信息臂）× 16 init（与 B29d 同 init×流，
逐 run 配对）× 3200 步，协议逐字同 B29c/B29d cx_v2。两处改动：
  1. 池化层：ComplexAMC2(pooling="orbit"|"hybrid"|"gated")。
  2. 评估：eval_cell_paired——三头评估对 σ∈{0,π/3,π} 用同一采样器种子，
     base episodes 逐位相同（注入在采样后叠加）⇒ σ 平坦性成为精确等变证书。
诊断：equivariance_error、cancel_ratio（隐层相干）、embedding 相干度
（heads_on 返回的 phasemap EM mean|γ|）。
输出：logs/B30_orbit_pool.json（增量落盘）。
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
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio
from run_b19 import l1pca_proto, embed, heads_on, cnorm
from run_b29 import new_ema, ema_update, ema_loadable
from run_b29c import make_train_val_samplers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INIT_SEEDS = (7, 13, 17, 19, 29, 31, 41, 43, 47, 53, 59, 61, 71, 79, 83, 89)
SIG = (0.0, np.pi / 3, np.pi)
# 不变读出（嵌入精确全局相位不变）：原型层不能再做相位对齐（l1pca 会
# 对齐 R(τ) 的相位差结构=破坏信息，B30 教训在原型层重演）；用均值原型+欧氏距离。
INVARIANT_ARMS = {"autocorr", "power"}


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
    """σ 配对三头评估：同一采样器种子 ⇒ base episodes 逐位相同，注入后叠加。
    σ 平坦性残差只来自 float 噪声（精确等变证书）。"""
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
        out[f"gamma_mag_{st:.2f}"] = h["gamma_mag"]          # 支持集对齐相干度（机制诊断）
    return out


def train_one(arm, i0, zn, y, snr, sp, budget, val_every, val_n, epi_eval,
              lam=1.0):
    torch.manual_seed(i0); np.random.seed(i0)
    trunk = ComplexAMC2(init="hann", pooling=arm, ch=(48, 96, 192, 48),
                        orb_lam=lam).to(DEV)
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
            "equiv_delta": float(equivariance_error(trunk, probe, device=DEV)),
            "cancel_ratio": float(cancellation_ratio(trunk, probe)),
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:])),
            "params": int(trunk.n_params()), "train_min": (time.time() - t0) / 60}
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
        accs = {s: 100 * ev[name][f"sigma_{s:.2f}"]["orbital"] for s in SIG}
        print(f"  [{arm}|i{i0}|{name}] " +
              " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
              f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
        del twin
    return info, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--arms", default="gated,orbit,hybrid")
    ap.add_argument("--lam", type=float, default=1.0, help="orbit 部分对齐强度（B26 收缩族旋钮；仅 orbit/hybrid 臂生效）")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]

    arms = a.arms.split(",")
    if a.mode == "smoke":
        arms, budget, val_every, val_n, epi_eval = tuple(a.arms.split(",")), 60, 20, 10, 20
    else:
        budget, val_every, val_n, epi_eval = 3200, 100, 100, a.epi_eval

    res_path = os.path.join(a.out, "logs", "B30_orbit_pool.json")
    recs = {"meta": {"prereg": "PREREG_B30.md", "arms": arms, "budget": budget,
                     "inits": list(INIT_SEEDS), "stream_seed": 4000,
                     "eval": "sigma-paired (same base episodes across sigma)",
                     "disjoint": True, "deterministic": True}, "runs": {}}
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        recs["runs"].update(old.get("runs", {}))
        print(f"[resume] {len(recs['runs'])} done", flush=True)
    t00 = time.time()
    for arm in arms:
        for i0 in INIT_SEEDS:
            tag = f"{arm}L{a.lam:g}" if a.lam != 1.0 else arm
            key = f"{tag}_i{i0}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            print(f"[run] {key}", flush=True)
            info, ev = train_one(arm, i0, zn, y, snr, sp, budget, val_every,
                                 val_n, epi_eval, lam=a.lam)
            recs["runs"][key] = {"info": info, "eval": ev}
            json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B30[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
