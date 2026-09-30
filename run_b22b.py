"""B22b：SC-A 翻正第二尝试——训练对齐器与评估侧严格一致（PREREG_B22 修正案 v2）。

H-game：B22 的回归源于训练目标（5 迭代单重启对齐）与评估侧 orbital_align
（3 重启×10 迭代取最优）不一致，长训练放大博弈。strict 臂把训练对齐器换成
评估侧的 torch 等价（对齐相位 stop-gradient，目标值选择不变量 ⇒ 等变保持）。
输出：logs/B22b_strictalign.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import RealAMC, unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2
from run_b17 import AmcCNNWrap
from run_b19 import train_protonet_hist, embed, heads_on

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def l1pca_strict(es, n_iter=10):
    """estim.orbital_align 的 torch 等价：3 确定性重启 × n_iter，按不变目标选最优。
    es:(B,N,k,C) complex。对齐相位 stop-gradient（EM 式）。"""
    k = es.shape[2]
    inits = [es.mean(2), es[:, :, 0], es[:, :, min(2, k - 1)]]
    best_mu, best_obj = None, None
    for mu0 in inits:
        mu = mu0
        for _ in range(n_iter):
            ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
            w = torch.exp(-1j * torch.angle(ip)).detach()
            mu = (es * w[..., None]).mean(2)
        obj = torch.einsum("bnc,bnkc->bnk", mu.conj(), es).abs().sum(-1)
        if best_obj is None:
            best_mu, best_obj = mu, obj
        else:
            better = obj > best_obj
            best_mu = torch.where(better[..., None], mu, best_mu)
            best_obj = torch.where(better, obj, best_obj)
    return best_mu


def train_l1pca(trunk, sampler, n_ep, strict=False, lr=1e-3, bs=8, device=DEV,
                log_every=400, tag=""):
    import torch.nn.functional as F
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    hist = []
    t0 = time.time()
    for it in range(n_ep):
        Zs, Zq, yq = sampler.sample(bs)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        es = trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1)
        eq = trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1)
        es, eq = unit_norm(es), unit_norm(eq)
        mu = es.mean(2) if not strict else None
        if strict:
            mu = l1pca_strict(es)
        else:
            for _ in range(5):
                ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
                w = torch.exp(-1j * torch.angle(ip)).detach()
                mu = (es * w[..., None]).mean(2)
        mu = unit_norm(mu)
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.item()))
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    trunk.eval()
    return trunk, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    budgets = {"cx1200_l1pca5": 1200, "cx1200_strict": 1200, "cx3200_strict": 3200}
    if a.mode == "smoke":
        budgets = {k: 40 for k in budgets}
        a.epi_eval, a.sigmas = 20, "0"
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000, snr_min=6, snr_max=18)
    recs = {"meta": {}, "train_hist": {}, "records": []}
    t00 = time.time()
    trunks = {}
    for name, bud in budgets.items():
        torch.manual_seed(11); np.random.seed(11)
        trunk = ComplexAMC2(init="hann", pooling="gated").to(DEV)
        trunk, hist = train_l1pca(trunk, smp_tr, bud,
                                  strict=name.endswith("_strict"),
                                  log_every=max(bud // 6, 50), tag=name)
        trunks[name] = trunk
        recs["train_hist"][name] = hist[::20]
        recs["meta"][name] = {"params": int(trunk.n_params()), "budget": bud,
                              "loss_first100": float(np.mean(hist[:100])),
                              "loss_last100": float(np.mean(hist[-100:]))}
        probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
        recs["meta"][name]["equiv_delta"] = equivariance_error(trunk, probe, device=DEV)
        print(f"  [{name}] budget={bud} loss {recs['meta'][name]['loss_first100']:.3f} -> "
              f"{recs['meta'][name]['loss_last100']:.3f} delta={recs['meta'][name]['equiv_delta']:.5f}",
              flush=True)
    for st in sigmas:
        smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                q_per_class=15, seed=7100 + int(97 * st),
                                snr_min=6, snr_max=18)
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp_te.sample(a.epi_eval, inject=inj)
        cell = {"sigma": float(st), "n_eval": a.epi_eval, "acc": {}}
        for name, tr in trunks.items():
            es = embed(tr, Zs).astype(np.complex128)
            eq = embed(tr, Zq).astype(np.complex128)
            h, _ = heads_on(es, eq, yq, st, unit=True)
            for k2, v in h.items():
                cell["acc"][f"{name}_{k2}"] = v
        recs["records"].append(cell)
        summ = {k2: 100 * float(np.mean(v)) for k2, v in cell["acc"].items()}
        print(f"  [σ={st:.2f}] " + " ".join(f"{k2}={v:.1f}" for k2, v in sorted(summ.items())),
              flush=True)
    json.dump(recs, open(os.path.join(a.out, "logs", "B22b_strictalign.json"), "w"), indent=1)
    print(f"B22b[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
