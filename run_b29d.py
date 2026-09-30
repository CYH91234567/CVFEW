"""B29d：早期盆筛选协议验证——cancel_ratio@800 训练时诊断（PREREG_B29d）。

16 新 init × 流 seed 4000，与 B29c cx_v2 逐字同协议；唯一增加：每 100 步记录
cancel_ratio 与 vt（训练时诊断，无标签/无 val 采样损耗）。val_train 选点照旧。
输出：logs/B29d_screening.json（增量落盘）。
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
from cvfe.nets_b17 import unit_norm
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio
from run_b19 import l1pca_proto, embed, heads_on, cnorm
from run_b29 import eval_cell, SIGMAS, new_ema, ema_update, ema_loadable
from run_b29c import make_train_val_samplers

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INIT_SEEDS = (7, 13, 17, 19, 29, 31, 41, 43, 47, 53, 59, 61, 71, 79, 83, 89)


def quick_val_train(trunk, val_smps, n):
    trunk.eval()
    vals = []
    for smp, st in val_smps:
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp.sample(n, inject=inj)
        es = cnorm(embed(trunk, Zs, device=DEV)).astype(np.complex128)
        eq = cnorm(embed(trunk, Zq, device=DEV)).astype(np.complex128)
        pred = E.cls_orbital(eq, E.proto_orbital(es))
        vals.append(float((pred == yq).mean()))
    trunk.train()
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]

    inits, budget = (INIT_SEEDS if a.mode == "full" else (7,)), (3200 if a.mode == "full" else 60)
    epi_eval = a.epi_eval if a.mode == "full" else 20

    res_path = os.path.join(a.out, "logs", "B29d_screening.json")
    recs = {"meta": {"prereg": "PREREG_B29d.md", "budget": budget,
                     "inits": list(inits), "stream_seed": 4000,
                     "disjoint": True, "deterministic": True}, "runs": {}}
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        recs["runs"].update(old.get("runs", {}))
        print(f"[resume] {len(recs['runs'])} done", flush=True)
    t00 = time.time()
    import torch.nn.functional as F
    for i0 in inits:
        key = f"cx_v2_d_i{i0}"
        if key in recs["runs"]:
            print(f"[skip] {key}", flush=True)
            continue
        torch.manual_seed(i0); np.random.seed(i0)
        trunk = ComplexAMC2(init="hann", pooling="gated", ch=(48, 96, 192, 48)).to(DEV)
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=4000, snr_min=6, snr_max=18)
        val_smps = make_train_val_samplers(zn, y, snr, sp, 88000)
        opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
        trunk.train()
        diag = []       # [(it, cancel_ratio, vt)]
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
            mu = unit_norm(l1pca_proto(es, 5))
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
            d2 = 2.0 - 2.0 * ip.abs()
            loss = F.cross_entropy(-d2.reshape(-1, N),
                                   torch.as_tensor(yq.reshape(-1), device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(ema_state, trunk)
            hist.append(float(loss.item()))
            if (it + 1) % 100 == 0:
                cr = cancellation_ratio(trunk, probe)
                vt = quick_val_train(trunk, val_smps, 100)
                diag.append({"it": it + 1, "cancel": cr, "vt": vt})
                if vt > best_val:
                    best_val, best_it = vt, it + 1
                    best_state = {kk: vv.detach().cpu().clone()
                                  for kk, vv in trunk.state_dict().items()}
                if (it + 1) % 400 == 0:
                    print(f"    [{key}] ep{it+1}/{budget} loss={loss.item():.4f} "
                          f"cr={cr:.3f} vt={vt:.3f} ({(time.time()-t0)/60:.1f}min)",
                          flush=True)
        trunk.eval()
        endpoint_state = {kk: vv.detach().cpu().clone()
                          for kk, vv in trunk.state_dict().items()}
        entry = {"info": {"best_val": best_val, "best_it": best_it,
                          "cancel_endpoint": cancellation_ratio(trunk, probe),
                          "loss_first100": float(np.mean(hist[:100])),
                          "loss_last100": float(np.mean(hist[-100:])),
                          "diag": diag, "params": int(trunk.n_params())},
                 "eval": {}}
        sets = {"endpoint": endpoint_state}
        if best_state is not None:
            sets["valsel"] = best_state
        if ema_state["state"] is not None:
            sets["ema"] = ema_loadable(ema_state, trunk)
        for name, sd in sets.items():
            twin = copy.deepcopy(trunk)
            twin.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
            twin.eval()
            entry["eval"][name] = eval_cell(twin, zn, y, snr, sp["test"], SIGMAS, epi_eval)
            accs = {s: 100 * entry["eval"][name][f"sigma_{s:.2f}"]["orbital"]
                    for s in SIGMAS}
            print(f"  [{key}|{name}] " +
                  " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
                  f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
            del twin
        recs["runs"][key] = entry
        json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B29d[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
