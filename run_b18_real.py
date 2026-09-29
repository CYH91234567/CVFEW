"""B18-real：真实 IQ 上"规范化（选截面）"的割迹代价 —— 自然精度差 Δ。

背景（B17 pilot 已证）：RML2016.10a 自然数据本身含近均匀相位 nuisance ⇒
在自然数据上训练的网络**已经**对注入相位免疫（98.9→98.6）。故 σ 轴无判别力，
判别力来自：规范化是否"免费"（自然精度差 Δ）以及在难格（0 dB / 难划分）上的代价。

臂（编码器统一 AmcCNNWrap = B10 SOTA 骨架，预算一致）：
  raw_nat   自然数据训练（对照）
  canon_nat 规范化后训练（逐 episode 支持集池化二阶矩主方向；严格 U(1) 不变）
  raw_aug   自然数据 + 随机相位旋转增广（Kumar 式最佳实践）
头（冻结编码器）：euclid / orbital / phasemap_ml；另报 identity（零训练）与规范化稳定性诊断。
输出：logs/B18_real.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm
from run_b10 import AmcCNN
from run_b17 import AmcCNNWrap, embed, heads_on, DEV

CANON_ARM = {"raw_nat": False, "canon_nat": True, "raw_aug": False}


def canon_episode(Zs, Zq):
    """按支持集池化二阶矩主方向规范化（严格 U(1) 不变）。"""
    v = E.canon_ref_vec(Zs.astype(np.complex128))
    return E.canon_apply_ref(Zs.astype(np.complex128), v), E.canon_apply_ref(
        Zq.astype(np.complex128), v), v


def align_strength(Zs, v):
    """|z^H v| / (||z||·||v||)：离割迹的距离（1=远离，0=在割迹上）。"""
    Zn = Zs.reshape(Zs.shape[0], -1, Zs.shape[-1]).astype(np.complex128)
    num = np.abs(np.einsum("eip,ep->ei", Zn, v))
    den = np.linalg.norm(Zn, axis=-1) * np.linalg.norm(v, axis=-1, keepdims=True)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def train_arm(trunk, smp_tr, n_ep, canon=False, aug=False, lr=1e-3, bs=8, tag=""):
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    t0 = time.time()
    for it in range(n_ep):
        Zs, Zq, yq = smp_tr.sample(bs, inject=None)
        if canon:
            Zs, Zq, _ = canon_episode(Zs, Zq)
            Zs, Zq = Zs.astype(np.complex64), Zq.astype(np.complex64)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        if aug:
            th = torch.exp(1j * 2 * np.pi * torch.rand(B * N * k, 1, device=DEV))
            Zs_t = (Zs_t.reshape(-1, L) * th).reshape(B, N, k, L)
            thq = torch.exp(1j * 2 * np.pi * torch.rand(B * m, 1, device=DEV))
            Zq_t = (Zq_t.reshape(-1, L) * thq).reshape(B, m, L)
        es = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1))
        mu = es.mean(2)
        d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 1000 == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    trunk.eval()
    return trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--snrs", default="0,12")
    ap.add_argument("--sigmas", default="0,3.14")
    ap.add_argument("--arms", default="raw_nat,canon_nat,raw_aug")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=a.splits)
    snrs = [int(s) for s in a.snrs.split(",")]
    sigmas = [float(s) for s in a.sigmas.split(",")]
    arms = a.arms.split(",")
    recs, meta = [], {}
    t0 = time.time()
    for si, sp in enumerate(splits):
        print(f"=== split {si} test={sp['test']} ===", flush=True)
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=4000 + si, snr_min=0, snr_max=18)
        trunks = {}
        for arm in arms:
            torch.manual_seed(1000 * si)
            trunks[arm] = train_arm(AmcCNNWrap().to(DEV), smp_tr, a.budget,
                                    canon=CANON_ARM.get(arm, False), aug=(arm == "raw_aug"),
                                    tag=f"{arm}")
            meta.setdefault(arm, {"params": int(trunks[arm].n_params())})
        for sl in snrs:
            for st in sigmas:
                smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                        q_per_class=15, seed=7100 + si + 7 * sl + int(97 * st),
                                        snr_min=sl, snr_max=sl)
                Zs, Zq, yq = smp_te.sample(a.epi_eval,
                                           inject=None if st == 0 else ("global", st))
                cell = {"split": si, "snr": sl, "sigma": float(st), "n_eval": a.epi_eval,
                        "acc": {}, "align": {}}
                # identity（零训练）：raw 与 canon 两种输入
                h, k_ = heads_on(Zs.astype(np.complex128), Zq.astype(np.complex128), yq, st, unit=True)
                for k2, v in h.items():
                    cell["acc"][f"ident_{k2}"] = v.tolist()
                Zsc, Zqc, v = canon_episode(Zs, Zq)
                cell["align"]["ident"] = align_strength(Zs, v)
                hc, _ = heads_on(Zsc, Zqc, yq, st, unit=True)
                for k2, v2 in hc.items():
                    cell["acc"][f"ident_canon_{k2}"] = v2.tolist()
                # 各臂
                for arm, tr in trunks.items():
                    if CANON_ARM.get(arm, False):
                        Zs_a, Zq_a = Zsc, Zqc
                    else:
                        Zs_a, Zq_a = Zs.astype(np.complex128), Zq.astype(np.complex128)
                    es = embed(tr, Zs_a).astype(np.complex128)
                    eq = embed(tr, Zq_a).astype(np.complex128)
                    hh, kap = heads_on(es, eq, yq, st, unit=True)
                    for k2, v3 in hh.items():
                        cell["acc"][f"{arm}_{k2}"] = v3.tolist()
                    cell[f"kappa_{arm}"] = kap
                recs.append(cell)
                summ = {k2: 100 * float(np.mean(v)) for k2, v in cell["acc"].items()
                        if k2.endswith(("euclid", "orbital", "phML"))}
                print(f"  [s{si} SNR{sl} σ={st:.2f}] " +
                      " ".join(f"{k2}={v:.1f}" for k2, v in summ.items()), flush=True)
    json.dump({"meta": meta, "records": recs, "args": vars(a)},
              open(os.path.join(a.out, "logs", "B18_real.json"), "w"), indent=1)
    print(f"B18-real DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
