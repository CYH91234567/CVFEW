"""B27：B7 复访——l1pca 原型目标下的 orbital-vs-增广预算扫描（服务器 3090）。

PREREG_B27。与 run_b67.run_b7 严格同协议（ComplexTrunk、EpisodeSampler seed=555、
bs=4、train_sigmas 循环 {0,π/3,π} global 注入、AdamW lr=1e-3 wd=1e-4、预算
{200,800,3200} × 种子 {0,1,2}），仅跑两个臂：
  orbital        朴素均值原型目标（B7 对照臂重跑，验证环境复现）
  orbital_l1pca  l1pca_proto(es,5) 原型目标（本轮新臂，B24 坍缩定理的修复）
评估：测试类 3-way，注入 {0,π/3,π}，200 epi；主指标 = B7 原判据（均值原型欧氏
d2 argmin）+ 辅助 orbital 头（estim.proto_orbital + cls_orbital）。
输出：logs/B27_b7revisit.json（每 run 增量落盘，断点续跑）。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.torch_nets import ComplexTrunk
from run_b19 import l1pca_proto

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGETS = (200, 800, 3200)
SEEDS = (0, 1, 2)
MODES = ("orbital", "orbital_l1pca")
TRAIN_SIGMAS = (0.0, np.pi / 3, np.pi)


def proto_loss(trunk, Zs_e, Zq_e, yq_e, mode, n_way, device):
    B, N, k, L = Zs_e.shape
    es = trunk(Zs_e.reshape(-1, L)).reshape(B, N, k, -1)
    eq = trunk(Zq_e.reshape(-1, L)).reshape(B, -1, es.shape[-1])
    if mode == "orbital_l1pca":
        mu = l1pca_proto(es, 5)
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = (eq.abs() ** 2).sum(-1).unsqueeze(-1) \
            + (mu.abs() ** 2).sum(-1).unsqueeze(1) - 2 * ip.abs()
    else:
        mu = es.mean(2)
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = (eq.abs() ** 2).sum(-1).unsqueeze(-1) \
            + (mu.abs() ** 2).sum(-1).unsqueeze(1) - 2 * ip.abs()
    return F.cross_entropy(-d2.reshape(-1, n_way),
                           torch.as_tensor(yq_e, dtype=torch.long, device=device).reshape(-1))


def eval_trunk(trunk, smp_te_base, sigmas=(0.0, np.pi / 3, np.pi), epi=200):
    out = {}
    for ev in sigmas:
        smp = smp_te_base
        Zs, Zq, yq = smp.sample(epi, inject=("global", float(ev)))
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            B, N, k, L = Zs_t.shape
            es_full = trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1).cpu().numpy()
            eq = trunk(Zq_t.reshape(-1, L)).reshape(Zq_t.shape[0], -1, es_full.shape[-1]).cpu().numpy()
        es_mean = es_full.mean(2).astype(np.complex128)
        eq_c = eq.astype(np.complex128)
        d2 = (np.abs(eq_c[:, :, None, :] - es_mean[:, None, :, :]) ** 2).sum(-1)
        out[f"meanproto_{ev:.2f}"] = float((d2.argmin(-1) == yq).mean())
        es_u = es_full.astype(np.complex128)
        es_u = es_u / np.maximum(np.linalg.norm(es_u, axis=-1, keepdims=True), 1e-12)
        eq_u = eq_c / np.maximum(np.linalg.norm(eq_c, axis=-1, keepdims=True), 1e-12)
        pred = E.cls_orbital(eq_u, E.proto_orbital(es_u))
        out[f"orbital_{ev:.2f}"] = float((pred == yq).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    budgets, seeds, modes = BUDGETS, SEEDS, MODES
    if a.mode == "smoke":
        budgets, seeds = (30,), (0,)
    res_path = os.path.join(a.out, "logs", "B27_b7revisit.json")
    recs = {"runs": []}
    if os.path.exists(res_path):
        recs = json.load(open(res_path))
        print(f"[resume] {len(recs['runs'])} runs done", flush=True)
    done = {(r["budget"], r["mode"], r["seed"]) for r in recs["runs"]}
    t00 = time.time()
    for budget in budgets:
        for mode in modes:
            for seed in seeds:
                if (budget, mode, seed) in done:
                    continue
                torch.manual_seed(seed)
                trunk = ComplexTrunk().to(DEV)
                opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
                smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                        k_shot=5, q_per_class=5, seed=555,
                                        snr_min=6, snr_max=18)
                t0 = time.time()
                hist = []
                for it in range(budget):
                    sg = TRAIN_SIGMAS[it % len(TRAIN_SIGMAS)]
                    Zs, Zq, yq = smp_tr.sample(4, inject=("global", float(sg)))
                    Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
                    Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
                    Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    loss = proto_loss(trunk, Zs_t, Zq_t, yq, mode, 5, DEV)
                    opt.zero_grad(); loss.backward(); opt.step()
                    hist.append(float(loss.item()))
                smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3,
                                        k_shot=5, q_per_class=15, seed=999 + seed,
                                        snr_min=6, snr_max=18)
                acc = eval_trunk(trunk, smp_te)
                recs["runs"].append({"budget": budget, "mode": mode, "seed": seed,
                                     "acc": acc,
                                     "loss_first50": float(np.mean(hist[:50])),
                                     "loss_last50": float(np.mean(hist[-50:]))})
                json.dump(recs, open(res_path, "w"), indent=1)
                print(f"[b27] budget={budget} mode={mode} seed={seed} "
                      f"loss {recs['runs'][-1]['loss_first50']:.3f}->"
                      f"{recs['runs'][-1]['loss_last50']:.3f}: " +
                      " ".join(f"{kk}={100*vv:.1f}" for kk, vv in sorted(acc.items())) +
                      f" ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"B27[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
