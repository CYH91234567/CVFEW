"""B29b：模型选择信号校准诊断（PREREG_B29b，服务器 3090）。

3 条轨迹 × 每 100 步 checkpoint × {test(诊断), val_5shot, val_1shot, val_train}。
目的：测量各选点信号与测试性能的轨迹内对齐度（Spearman）+ 盆存在性
（oracle checkpoint 上界）。测试评估仅作诊断读数，不作模型选择（B29c 终版
判定用全新 init 种子）。
输出：logs/B29b_valsel_calibration.json（增量落盘）。
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
from cvfe.nets_b19 import ComplexAMC2
from run_b19 import l1pca_proto, embed, heads_on, cnorm
from run_b29 import make_val_samplers, SIGMAS, new_ema, ema_update

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAJECTORIES = [(11, 0), (11, 12), (23, 0)]


@torch.no_grad()
def quick_head_acc(trunk, smp, head, n_ep, inject, device=DEV):
    trunk.eval()
    Zs, Zq, yq = smp.sample(n_ep, inject=inject)
    es = cnorm(embed(trunk, Zs, device=device)).astype(np.complex128)
    eq = cnorm(embed(trunk, Zq, device=device)).astype(np.complex128)
    if head == "orbital":
        pred = E.cls_orbital(eq, E.proto_orbital(es))
    elif head == "euclid":
        pred = E.cls_euclid(eq, E.proto_euclid(es))
    else:
        raise KeyError(head)
    return float((pred == yq).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--epi-test", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    res_path = os.path.join(a.out, "logs", "B29b_valsel_calibration.json")
    recs = {"meta": {"prereg": "PREREG_B29b.md", "trajectories": TRAJECTORIES,
                     "epi_test": a.epi_test, "n_val": a.n_val}, "trajs": {}}
    if os.path.exists(res_path):
        recs = json.load(open(res_path))
        print(f"[resume] {len(recs['trajs'])} trajs done", flush=True)
    t00 = time.time()
    for init_seed, stream_off in TRAJECTORIES:
        key = f"i{init_seed}_f{stream_off}"
        if key in recs["trajs"]:
            print(f"[skip] {key}", flush=True)
            continue
        torch.manual_seed(init_seed); np.random.seed(init_seed)
        trunk = ComplexAMC2(init="hann", pooling="gated", ch=(48, 96, 192, 48)).to(DEV)
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=4000 + stream_off,
                                snr_min=6, snr_max=18)
        val5 = make_val_samplers(zn, y, snr, sp, 2, 5, 15, 99000)          # val_v1
        val1 = make_val_samplers(zn, y, snr, sp, 2, 1, 15, 88000)          # val_v2 (1-shot)
        # val_v3：训练类 5-way 5-shot（B25 旧 val 对照）——train 类有 6 个
        valtr = [(EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                 k_shot=5, q_per_class=5, seed=77000 + 1000 + i,
                                 snr_min=6, snr_max=18), st)
                 for i, st in enumerate(SIGMAS)]
        test_smps = []
        for st in SIGMAS:
            test_smps.append((EpisodeSampler(
                zn, y, snr, classes=sp["test"], n_way=3, k_shot=5, q_per_class=15,
                seed=7100 + int(97 * st), snr_min=6, snr_max=18), st))
        print(f"[{key}] training + per-checkpoint eval", flush=True)
        curve = []
        import torch.nn.functional as F
        opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
        ema_state = new_ema()
        trunk.train()
        t0 = time.time()
        for it in range(3200):
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
            if (it + 1) % 100 == 0:
                rec = {"it": it + 1}
                # 诊断测试（仅读数）
                tacc = []
                for smp, st in test_smps:
                    inj = None if st == 0 else ("global", st)
                    tacc.append(quick_head_acc(trunk, smp, "orbital", a.epi_test, inj))
                rec["test_orbital"] = float(np.mean(tacc))
                rec["test_by_sigma"] = [float(v) for v in tacc]
                # val 变体
                rec["val_5shot"] = float(np.mean([
                    quick_head_acc(trunk, smp, "orbital", a.n_val,
                                   None if st == 0 else ("global", st))
                    for smp, st in val5]))
                rec["val_1shot"] = float(np.mean([
                    quick_head_acc(trunk, smp, "orbital", a.n_val,
                                   None if st == 0 else ("global", st))
                    for smp, st in val1]))
                rec["val_train"] = float(np.mean([
                    quick_head_acc(trunk, smp, "orbital", 100,
                                   None if st == 0 else ("global", st))
                    for smp, st in valtr]))
                curve.append(rec)
                if (it + 1) % 400 == 0:
                    print(f"  [{key}] ck{it+1} test={rec['test_orbital']:.3f} "
                          f"v5={rec['val_5shot']:.3f} v1={rec['val_1shot']:.3f} "
                          f"vt={rec['val_train']:.3f} ({(time.time()-t0)/60:.1f}min)",
                          flush=True)
        # 轨迹内对齐
        tvec = np.array([r["test_orbital"] for r in curve])
        align = {}
        for vn in ("val_5shot", "val_1shot", "val_train"):
            vvec = np.array([r[vn] for r in curve])
            from scipy.stats import spearmanr, pearsonr
            align[vn] = {"spearman": float(spearmanr(vvec, tvec).statistic),
                         "pearson": float(pearsonr(vvec, tvec).statistic),
                         "max": float(vvec.max())}
        summary = {
            "align": align,
            "test_max": float(tvec.max()), "test_max_it": int(curve[int(np.argmax(tvec))]["it"]),
            "test_endpoint": float(tvec[-1]),
            "test_at_v5sel": float(tvec[int(np.argmax([r["val_5shot"] for r in curve]))]),
            "test_at_v1sel": float(tvec[int(np.argmax([r["val_1shot"] for r in curve]))]),
            "test_at_vtrsel": float(tvec[int(np.argmax([r["val_train"] for r in curve]))]),
            "curve": curve}
        recs["trajs"][key] = summary
        json.dump(recs, open(res_path, "w"), indent=1)
        print(f"[{key}] test_max={summary['test_max']:.3f}@{summary['test_max_it']} "
              f"endpoint={summary['test_endpoint']:.3f} "
              f"v1sel={summary['test_at_v1sel']:.3f} "
              f"align=" + " ".join(f"{k}:{v['spearman']:.2f}" for k, v in align.items()),
              flush=True)
    print(f"B29b DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
