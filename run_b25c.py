"""B25c：SC-A 最终组合臂——big trunk × 混合 σ_θ 注入 × 早停（PREREG_B25c）。

臂：big_mix1200 / big_mix_es3200，种子 {11,23,37}，l1pca5 目标，
训练 episode 按 {σ_θ=0,π/3,π} 循环 global 注入，eval 与 B25 同协议。
输出：logs/B25c_combo.json。
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, fisher_ratio
from run_b19 import l1pca_proto, embed, heads_on
from run_b25 import train_arm, eval_cell

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (11, 23, 37)
SEED_STREAM_OFF = {11: 0, 23: 12, 37: 26}
TRAIN_SIGMAS = (0.0, np.pi / 3, np.pi)
ARMS = {
    "big_mix1200":    dict(ch=(48, 96, 192, 48), budget=1200, es=False),
    "big_mix_es3200": dict(ch=(48, 96, 192, 48), budget=3200, es=True),
}


def train_mix(trunk, smp_tr, n_ep, es, bs=8, device=DEV, tag="", log_every=400,
              es_every=100, es_patience=5, es_min=600, val_n=100, val_sampler=None):
    """l1pca5 目标 + 混合 σ_θ 注入训练（复用 run_b25.train_arm 的早停框架，
    在 sample 处注入）。"""
    import torch.nn.functional as F
    opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
    trunk.train()
    hist, val_curve = [], []
    best_val, best_state, best_it, bad, stopped = -1.0, None, 0, 0, n_ep
    t0 = time.time()
    for it in range(n_ep):
        inj = ("global", float(TRAIN_SIGMAS[it % len(TRAIN_SIGMAS)]))
        Zs, Zq, yq = smp_tr.sample(bs, inject=inj)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        es_emb = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, -1, es_emb.shape[-1]))
        mu = unit_norm(l1pca_proto(es_emb, 5))
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.item()))
        if es and (it + 1) % es_every == 0:
            from run_b25 import val_orbital_acc
            was_training = trunk.training
            v = val_orbital_acc(trunk, val_sampler, val_n, device)
            if not was_training:
                trunk.eval()
            else:
                trunk.train()
            val_curve.append([it + 1, v])
            if v > best_val:
                best_val, best_it, bad = v, it + 1, 0
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
            else:
                bad += 1
            if (it + 1) >= es_min and bad >= es_patience:
                stopped = it + 1
                print(f"    [{tag}] early stop @{it+1} best_val={best_val:.3f}@{best_it}",
                      flush=True)
                break
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    if es and best_state is not None:
        trunk.load_state_dict(best_state)
    trunk.eval()
    info = {"stopped_at": stopped if es else n_ep,
            "best_val": best_val, "best_it": best_it, "val_curve": val_curve,
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:]))}
    return trunk, hist, info


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
    arms, seeds = ARMS, SEEDS
    if a.mode == "smoke":
        arms = {"big_mix1200": ARMS["big_mix1200"]}
        seeds, a.epi_eval, sigmas = (11,), 20, [0.0]
        arms["big_mix1200"]["budget"] = 40
    res_path = os.path.join(a.out, "logs", "B25c_combo.json")
    recs = {"runs": {}}
    if os.path.exists(res_path):
        recs = json.load(open(res_path))
        print(f"[resume] {len(recs['runs'])} done", flush=True)
    t00 = time.time()
    for arm, cfg in arms.items():
        for seed in seeds:
            key = f"{arm}_s{seed}"
            if key in recs["runs"]:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            trunk = ComplexAMC2(init="hann", pooling="gated", ch=cfg["ch"]).to(DEV)
            smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                    k_shot=5, q_per_class=5,
                                    seed=4000 + SEED_STREAM_OFF[seed],
                                    snr_min=6, snr_max=18)
            val_smp = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                     k_shot=5, q_per_class=5, seed=99000 + seed,
                                     snr_min=6, snr_max=18) if cfg["es"] else None
            trunk, hist, info = train_mix(trunk, smp_tr, cfg["budget"], cfg["es"],
                                          val_sampler=val_smp, tag=key)
            probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
            info["equiv_delta"] = equivariance_error(trunk, probe, device=DEV)
            cell = eval_cell(trunk, zn, y, snr, sp["test"], sigmas, a.epi_eval)
            recs["runs"][key] = {"info": info, "eval": cell, "params": int(trunk.n_params())}
            for st in sigmas:
                acc = cell.get(f"sigma_{st:.2f}", {})
                print(f"  [{key}] σ={st:.2f}: " +
                      " ".join(f"{kk}={100*vv:.1f}" for kk, vv in sorted(acc.items())),
                      flush=True)
            json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B25c[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
