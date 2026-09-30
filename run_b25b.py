"""B25b：SC-A 方差攻击——EMA / bs16 / 混合 σ_θ 注入（PREREG_B25b，服务器 3090）。

B25 中期发现：base1200 三种子轨道头均值 90.9/70.6/79.0（sd≈10pt）——跨流方差是
SC-A 的主约束。四臂（全部 ComplexAMC2 hann+gated base，l1pca5 @1200，种子 {11,23,37}）：
  ema1200         参数 EMA(0.999，含 ModBN running) 评估 + 原始端点一并报告
  bs16_1200       bs=16（episode 数不变）
  mixsig_1200     训练 episode 按 {σ_θ=0,π/3,π} 循环 global 注入（B7 协议）
  mixsig_ema1200  mix + EMA
输出：logs/B25b_variance.json（增量落盘，断点续跑）。
"""
import argparse, json, os, sys, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, fisher_ratio
from run_b19 import l1pca_proto, embed, heads_on

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (11, 23, 37)
SEED_STREAM_OFF = {11: 0, 23: 12, 37: 26}
ARMS = {
    "ema1200":        dict(ema=True,  bs=8,  mix=False),
    "bs16_1200":      dict(ema=False, bs=16, mix=False),
    "mixsig_1200":    dict(ema=False, bs=8,  mix=True),
    "mixsig_ema1200": dict(ema=True,  bs=8,  mix=True),
}
TRAIN_SIGMAS = (0.0, np.pi / 3, np.pi)


def new_ema(trunk, decay=0.999):
    ema = {k: v.detach().clone().float() if v.is_floating_point()
           else v.detach().clone() for k, v in trunk.state_dict().items()}
    return ema, decay


def ema_update(ema, trunk, decay):
    with torch.no_grad():
        for k, v in trunk.state_dict().items():
            if v.is_floating_point():
                ema[k].mul_(decay).add_(v.detach().float(), alpha=1 - decay)
            else:
                ema[k] = v.detach().clone()


def train_arm(trunk, smp_tr, n_ep, bs=8, mix=False, ema=False, lr=1e-3,
              device=DEV, log_every=400, tag=""):
    import torch.nn.functional as F
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    hist, ema_state = [], (new_ema(trunk) if ema else None)
    t0 = time.time()
    for it in range(n_ep):
        inject = ("global", float(TRAIN_SIGMAS[it % len(TRAIN_SIGMAS)])) if mix else None
        Zs, Zq, yq = smp_tr.sample(bs, inject=inject)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        es = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, -1, es.shape[-1]))
        mu = unit_norm(l1pca_proto(es, 5))
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.item()))
        if ema:
            ema_update(ema_state[0], trunk, ema_state[1])
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    trunk.eval()
    return trunk, hist, (ema_state[0] if ema else None)


def eval_cell(trunk, zn, y, snr, sp_test, sigmas, epi_eval):
    out = {}
    for st in sigmas:
        smp_te = EpisodeSampler(zn, y, snr, classes=sp_test, n_way=3, k_shot=5,
                                q_per_class=15, seed=7100 + int(97 * st),
                                snr_min=6, snr_max=18)
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp_te.sample(epi_eval, inject=inj)
        es = embed(trunk, Zs).astype(np.complex128)
        eq = embed(trunk, Zq).astype(np.complex128)
        h, kap = heads_on(es, eq, yq, st, unit=True)
        out[f"sigma_{st:.2f}"] = {kk: float(np.mean(v)) for kk, v in h.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    arms, seeds, budget = ARMS, SEEDS, 1200
    if a.mode == "smoke":
        arms = {k: v for k, v in ARMS.items() if k in ("ema1200", "mixsig_ema1200")}
        seeds, budget, a.epi_eval, sigmas = (11,), 40, 20, [0.0]
    res_path = os.path.join(a.out, "logs", "B25b_variance.json")
    recs = {"runs": {}}
    if os.path.exists(res_path):
        recs = json.load(open(res_path))
        print(f"[resume] {len(recs['runs'])} done", flush=True)
    t00 = time.time()
    for arm, cfg in arms.items():
        for seed in seeds:
            key = f"{arm}_s{seed}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            trunk = ComplexAMC2(init="hann", pooling="gated").to(DEV)
            smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                    k_shot=5, q_per_class=5,
                                    seed=4000 + SEED_STREAM_OFF[seed],
                                    snr_min=6, snr_max=18)
            trunk, hist, ema_state = train_arm(
                trunk, smp_tr, budget, bs=cfg["bs"], mix=cfg["mix"],
                ema=cfg["ema"], tag=key)
            probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
            info = {"equiv_delta": equivariance_error(trunk, probe, device=DEV),
                    "loss_last100": float(np.mean(hist[-100:]))}
            cell = eval_cell(trunk, zn, y, snr, sp["test"], sigmas, a.epi_eval)
            recs["runs"][key] = {"info": info, "eval": cell, "params": int(trunk.n_params())}
            if ema_state is not None:
                ema_trunk = copy.deepcopy(trunk)
                ema_trunk.load_state_dict({k: v.to(DEV) for k, v in ema_state.items()})
                ema_trunk.eval()
                cell_e = eval_cell(ema_trunk, zn, y, snr, sp["test"], sigmas, a.epi_eval)
                recs["runs"][key]["eval_ema"] = cell_e
                recs["runs"][key]["info"]["equiv_delta_ema"] = equivariance_error(
                    ema_trunk, probe, device=DEV)
                del ema_trunk
            for st in sigmas:
                acc = cell.get(f"sigma_{st:.2f}", {})
                print(f"  [{key}] σ={st:.2f}: " +
                      " ".join(f"{kk}={100*vv:.1f}" for kk, vv in sorted(acc.items())),
                      flush=True)
            if "eval_ema" in recs["runs"][key]:
                for st in sigmas:
                    acc = recs["runs"][key]["eval_ema"].get(f"sigma_{st:.2f}", {})
                    print(f"  [{key}][EMA] σ={st:.2f}: " +
                          " ".join(f"{kk}={100*vv:.1f}" for kk, vv in sorted(acc.items())),
                          flush=True)
            json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B25b[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
