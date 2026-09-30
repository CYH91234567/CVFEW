"""B29c：终版 SC-A——校准选点信号(val_train) + 12 runs 分布报告（PREREG_B29c）。

与 B29 cx_v2 唯一差别：选点信号从 2-way val 类（饱和，Spearman 0.14-0.65）
换成 **训练类 5-way val**（不饱和，Spearman 0.79-0.83，B29b 校准结论）。
runs = init {11,23,37,51,67,73} × 流 {0,12} = 12。
输出：logs/B29c_sca_v3.json（增量落盘）。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, sys, time
import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b19 import ComplexAMC2
from run_b29 import train_arm, eval_cell, SIGMAS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INIT_SEEDS = (11, 23, 37, 51, 67, 73)
STREAM_OFFS = (0, 12)


def make_train_val_samplers(zn, y, snr, sp, seed0):
    """val_train（B29b 校准版）：训练类 5-way 5-shot q=5，σ 注入均值。"""
    out = []
    for i, st in enumerate(SIGMAS):
        smp = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                             k_shot=5, q_per_class=5, seed=seed0 + 1000 + i,
                             snr_min=6, snr_max=18)
        out.append((smp, st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]

    inits, offs, budget = INIT_SEEDS, STREAM_OFFS, 3200
    val_every, val_n = 100, 100
    if a.mode == "smoke":
        inits, offs, budget, val_every, val_n, a.epi_eval, sigmas = \
            (11,), (0,), 60, 20, 10, 20, [0.0]

    res_path = os.path.join(a.out, "logs", "B29c_sca_v3.json")
    recs = {"meta": {"prereg": "PREREG_B29c.md", "budget": budget,
                     "val_every": val_every, "val_n": val_n,
                     "epi_eval": a.epi_eval, "inits": list(inits),
                     "stream_offs": list(offs), "val_signal": "train_5way",
                     "disjoint": True, "deterministic": True},
            "runs": {}}
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        recs["runs"].update(old.get("runs", {}))
        print(f"[resume] {len(recs['runs'])} runs done", flush=True)
    t00 = time.time()
    for i0 in inits:
        for so in offs:
            key = f"cx_v2_vt_i{i0}_f{so}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            torch.manual_seed(i0); np.random.seed(i0)
            trunk = ComplexAMC2(init="hann", pooling="gated",
                                ch=(48, 96, 192, 48)).to(DEV)
            smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                    k_shot=5, q_per_class=5,
                                    seed=4000 + so, snr_min=6, snr_max=18)
            val_smps = make_train_val_samplers(zn, y, snr, sp, 77000 + 1000 * so)
            print(f"[{key}] budget={budget} val=train_5way", flush=True)
            trunk, hist, info, best_state, ema_state = train_arm(
                trunk, smp_tr, budget, "l1pca5", val_every=val_every,
                val_smps=val_smps, val_head="orbital", val_n=val_n,
                tag=key, log_every=max(budget // 8, 20))
            from run_b29 import equivariance_error, cancellation_ratio, fisher_ratio
            from run_b29 import embed
            probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
            info["equiv_delta"] = equivariance_error(trunk, probe, device=DEV)
            info["cancel_ratio"] = cancellation_ratio(trunk, probe)
            endpoint_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
            entry = {"info": info, "hist": hist[::20],
                     "params": int(trunk.n_params()), "eval": {}}
            sets = {"endpoint": endpoint_state}
            if best_state is not None:
                sets["valsel"] = best_state
            from run_b29 import ema_loadable
            if ema_state is not None and ema_state["state"] is not None:
                sets["ema"] = ema_loadable(ema_state, trunk)
            for name, sd in sets.items():
                import copy
                twin = copy.deepcopy(trunk)
                twin.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
                twin.eval()
                entry["eval"][name] = eval_cell(twin, zn, y, snr, sp["test"],
                                                sigmas, a.epi_eval)
                accs = {s: 100 * entry["eval"][name][f"sigma_{s:.2f}"]["orbital"]
                        for s in sigmas}
                print(f"  [{key}|{name}] " +
                      " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
                      f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
                del twin
            recs["runs"][key] = entry
            json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B29c[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
