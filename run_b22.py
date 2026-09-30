"""B22：SC2 翻正——预算/容量/目标三因子（服务器 3090；PREREG_B22）。

臂：cx_l1pca / cx_soft(softγ, σ_θ=π/6) / cx_big_l1pca(48,96,192,48) / real / amc，
预算 3200，eval 与 B19 同协议（split0，3-way 测试类，σ_θ∈{0,π/3,π}，200 epi）。
softγ 目标：PhaseMAP E 步的 torch 版（固定 σ_θ 网格 γ，stop-gradient），μ̂=Σγ_j z_j/k。
输出：logs/B22_sc2.json。
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import RealAMC, unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, fisher_ratio
from run_b17 import AmcCNNWrap
from run_b19 import train_protonet_hist, embed, heads_on

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gamma_soft(ip, st, n_grid=64, n_img=6):
    """E[e^{-iθ}|z] 的 torch 网格版（wrapN(0,st²) 先验）。ip:(...) complex。"""
    th = torch.linspace(-np.pi, np.pi, n_grid, device=ip.device, dtype=ip.real.dtype)
    js = torch.arange(-n_img, n_img + 1, device=ip.device, dtype=ip.real.dtype)
    rho = 2.0 * ip.abs()
    ll = torch.exp(rho[..., None] * (torch.cos(ip.angle()[..., None] - th) - 1.0))
    imgs = th[None, :] + 2 * np.pi * js[:, None]                     # (J,G)
    pr = torch.exp(-0.5 * (imgs / st) ** 2).sum(0)
    pr = pr / pr.sum()
    post = ll * pr
    num = (post * torch.exp(-1j * th)).sum(-1)
    return num / post.sum(-1).clamp_min(1e-30)


def train_softgamma(trunk, sampler, n_ep, st, lr=1e-3, bs=8, device=DEV,
                    log_every=400, tag=""):
    """软 γ 原型目标（3 轮 E 步迭代，γ stop-gradient）+ 轨道读出。"""
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
        mu = es.mean(2)
        for _ in range(3):
            ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
            g = gamma_soft(ip, st).detach()
            mu = (es * g[..., None]).sum(2) / k
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), unit_norm(mu))
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
    ap.add_argument("--budget", type=int, default=3200)
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    if a.mode == "smoke":
        a.budget, a.epi_eval, a.sigmas = 40, 20, "0"
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000, snr_min=6, snr_max=18)
    recs = {"meta": {}, "train_hist": {}, "records": []}
    t00 = time.time()

    def mk(name):
        if name == "cx_l1pca":
            return ComplexAMC2(init="hann", pooling="gated").to(DEV)
        if name == "cx_soft":
            return ComplexAMC2(init="hann", pooling="gated").to(DEV)
        if name == "cx_big_l1pca":
            return ComplexAMC2(init="hann", pooling="gated",
                               ch=(48, 96, 192, 48)).to(DEV)
        if name == "real":
            return RealAMC().to(DEV)
        if name == "amc":
            return AmcCNNWrap().to(DEV)
        raise KeyError(name)

    arms = ["cx_l1pca", "cx_soft", "cx_big_l1pca", "real", "amc"]
    if a.mode == "smoke":
        arms = arms[:3]
    trunks = {}
    for name in arms:
        torch.manual_seed(11); np.random.seed(11)
        trunk = mk(name)
        if name == "cx_soft":
            trunk, hist = train_softgamma(trunk, smp_tr, a.budget, np.pi / 6, tag=name)
        else:
            obj = "l1pca" if name.startswith("cx") else "euclid"
            trunk, hist = train_protonet_hist(trunk, smp_tr, a.budget, obj,
                                              log_every=max(a.budget // 8, 50), tag=name)
        trunks[name] = trunk
        recs["train_hist"][name] = hist[::20]
        recs["meta"][name] = {"params": int(trunk.n_params()),
                              "loss_first100": float(np.mean(hist[:100])),
                              "loss_last100": float(np.mean(hist[-100:]))}
        print(f"  [{name}] params={recs['meta'][name]['params']} "
              f"loss {recs['meta'][name]['loss_first100']:.3f} -> "
              f"{recs['meta'][name]['loss_last100']:.3f}", flush=True)
        if name.startswith("cx"):
            probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
            recs["meta"][name]["equiv_delta"] = equivariance_error(trunk, probe, device=DEV)
    for st in sigmas:
        smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                q_per_class=15, seed=7100 + int(97 * st),
                                snr_min=6, snr_max=18)
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp_te.sample(a.epi_eval, inject=inj)
        cell = {"sigma": float(st), "budget": a.budget, "n_eval": a.epi_eval, "acc": {}}
        for name, tr in trunks.items():
            es = embed(tr, Zs).astype(np.complex128)
            eq = embed(tr, Zq).astype(np.complex128)
            h, kap = heads_on(es, eq, yq, st, unit=True)
            for k2, v in h.items():
                cell["acc"][f"{name}_{k2}"] = v
        recs["records"].append(cell)
        summ = {k2: 100 * float(np.mean(v)) for k2, v in cell["acc"].items()}
        print(f"  [σ={st:.2f}] " + " ".join(f"{k2}={v:.1f}" for k2, v in sorted(summ.items())),
              flush=True)
    json.dump(recs, open(os.path.join(a.out, "logs", "B22_sc2.json"), "w"), indent=1)
    print(f"B22[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
