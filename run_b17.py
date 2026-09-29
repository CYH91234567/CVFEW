"""B17：商空间分类头 × 学习到的表示（真实 IQ, RML2016.10a, 防泄漏协议）。

动机（B10 暴露的短板 + B16b 的负结果）：
  - B10：零训练 PhaseMAP(identity 特征) 56.9 vs SOTA ProtoNet-CNN 87.1~89.7 —— 差在特征空间；
  - B16b：tiny ComplexTrunk 端到端训练后 trunk 系 48.5 ≈ identity 49.9 —— 差在编码器容量；
    但其嵌入上仍复现定理：欧氏头随 σ_θ 掉 12.4pt，商头精确平坦。
  ⇒ B17：容量匹配的编码器 + 可插拔商空间分类头（零重训练），把定理搬到 SOTA 精度档。

编码器（全部输出复向量，U(1) 作用 = 乘 e^{iθ}）：
  cx      ComplexAMC（严格等变，ModBN+modReLU+均值池化，32 复维）
  real    RealAMC（同宽非等变实对照，32 复维视角，参数≈2×）
  amc     AmcCNN（B10 的 SOTA 骨架，包装为 64 复维视角）
  ident   identity（能量归一化原始 IQ，零训练参照）

训练目标（预算完全匹配，自然数据，只用 train 类）：
  euclid（标准原型损失）/ orbital（商目标 d²=2−2|⟨e,μ̂⟩|）/ euclid+旋转增广(aug)

评估头（冻结编码器，零重训练）：euclid / orbital / phML(κ̂轮廓) / phML(oracle κ)
压力轴：逐帧独立全局相位 σ_θ ∈ {0(自然), π/6, π/3, π/2, 2π/3, π}

预注册：见 02_plan/PREREG_B17.md。主终点 = 压力格(σ_θ≥π/3)平均精度。
输出：logs/B17_{stage}.json（逐 episode 精度数组，供配对检验）。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe import estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import ComplexAMC, RealAMC, unit_norm, equivariance_error
from run_b10 import AmcCNN

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class AmcCNNWrap(nn.Module):
    """把 B10 的 SOTA 实值骨架包装成复输出（64 复维视角），使各头可统一调用。"""
    def __init__(self):
        super().__init__()
        self.net = AmcCNN()
        self.out_dim = 64

    def forward(self, z):                       # (B,L) complex -> (B,64) complex
        h = self.net(z)                         # (B,128) real
        return torch.view_as_complex(h.reshape(h.shape[0], -1, 2))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================ 训练
def train_protonet(trunk, sampler, n_ep, objective="euclid", aug=False, unit=True,
                   lr=1e-3, bs=8, device=DEV, log_every=1000, tag=""):
    """标准 N-way K-shot 原型训练。objective: euclid|orbital。"""
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    t0 = time.time()
    for it in range(n_ep):
        Zs, Zq, yq = sampler.sample(bs, inject=None)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        if aug:
            th = torch.exp(1j * 2 * np.pi * torch.rand(B * N * k, 1, device=device))
            Zs_t = (Zs_t.reshape(-1, L) * th).reshape(B, N, k, L)
            thq = torch.exp(1j * 2 * np.pi * torch.rand(B * m, 1, device=device))
            Zq_t = (Zq_t.reshape(-1, L) * thq).reshape(B, m, L)
        es = trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1)
        eq = trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1)
        if unit:
            es, eq = unit_norm(es), unit_norm(eq)
        mu = es.mean(2)
        if objective == "euclid":
            d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        else:                                    # orbital：商目标（与欧氏同尺度 [0,4]）
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), unit_norm(mu))
            d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    trunk.eval()
    return trunk


# ============================================================ 嵌入与评估
def embed(trunk, Z, device=DEV, bs=1024):
    """Z:(...,L) complex -> (...,C) complex numpy。"""
    shape = Z.shape
    flat = Z.reshape(-1, shape[-1])
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            zt = torch.as_tensor(flat[i:i + bs], dtype=torch.complex64, device=device)
            outs.append(trunk(zt).cpu().numpy())
    return np.concatenate(outs, 0).reshape(shape[:-1] + (outs[0].shape[-1],))


def cnorm(Z):
    n = np.linalg.norm(Z, axis=-1, keepdims=True)
    return Z / np.maximum(n, 1e-12)


def acc_of(pred, yq):
    return (pred == yq).mean(1)


def heads_on(Zs, Zq, yq, sigma_true, unit=True):
    """在给定嵌入上跑全部头。返回 {name: 逐episode精度} 与 κ̂诊断。"""
    if unit:
        Zs, Zq = cnorm(Zs), cnorm(Zq)
    Zs = np.ascontiguousarray(Zs, dtype=np.complex128)
    Zq = np.ascontiguousarray(Zq, dtype=np.complex128)
    out = {}
    out["euclid"] = acc_of(E.cls_euclid(Zq, E.proto_euclid(Zs)), yq)
    out["orbital"] = acc_of(E.cls_orbital(Zq, E.proto_orbital(Zs)), yq)
    mu_p, aux_p = E.phasemap_em(Zs)
    out["phML"] = acc_of(E.cls_marginal(Zq, mu_p, aux_p), yq)
    kap = float(np.mean(aux_p["sigma_th"][:, 0]))
    mu_o, aux_o = E.phasemap_em(Zs, sigma_th=max(float(sigma_true), 1e-3))
    out["phML_oracle"] = acc_of(E.cls_marginal(Zq, mu_o, aux_o), yq)
    return out, kap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.57,3.14")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.stage == "pilot":
        a.splits = a.splits or 1
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=a.splits)
    t0 = time.time()
    recs = []
    meta = {}
    for si, sp in enumerate(splits):
        print(f"=== split {si} train={sp['train']} test={sp['test']} ===", flush=True)
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=4000 + si, snr_min=6, snr_max=18)
        for seed in range(a.seeds):
            torch.manual_seed(1000 * si + seed)
            np.random.seed(1000 * si + seed)
            trunks = {}
            # ---- 训练（预算完全匹配）----
            if a.stage == "pilot":
                trunks["amc_euclid_raw"] = train_protonet(
                    AmcCNNWrap().to(DEV), smp_tr, a.budget, "euclid", unit=False,
                    tag="amc/raw", log_every=500)
                trunks["amc_euclid"] = train_protonet(
                    AmcCNNWrap().to(DEV), smp_tr, a.budget, "euclid", unit=True,
                    tag="amc/unit", log_every=500)
                trunks["amc_aug"] = train_protonet(
                    AmcCNNWrap().to(DEV), smp_tr, a.budget, "euclid", aug=True, unit=True,
                    tag="amc/aug", log_every=500)
                trunks["real_euclid"] = train_protonet(
                    RealAMC().to(DEV), smp_tr, a.budget, "euclid", unit=True,
                    tag="real/unit", log_every=500)
                trunks["cx_orbital"] = train_protonet(
                    ComplexAMC().to(DEV), smp_tr, a.budget, "orbital", unit=True,
                    tag="cx/orb", log_every=500)
                trunks["cx_euclid"] = train_protonet(
                    ComplexAMC().to(DEV), smp_tr, a.budget, "euclid", unit=True,
                    tag="cx/euc", log_every=500)
            else:
                trunks["amc_euclid"] = train_protonet(
                    AmcCNNWrap().to(DEV), smp_tr, a.budget, "euclid", unit=True, tag="amc", log_every=1000)
                trunks["amc_aug"] = train_protonet(
                    AmcCNNWrap().to(DEV), smp_tr, a.budget, "euclid", aug=True, unit=True, tag="amcA", log_every=1000)
                trunks["real_euclid"] = train_protonet(
                    RealAMC().to(DEV), smp_tr, a.budget, "euclid", unit=True, tag="real", log_every=1000)
                trunks["real_aug"] = train_protonet(
                    RealAMC().to(DEV), smp_tr, a.budget, "euclid", aug=True, unit=True, tag="realA", log_every=1000)
                trunks["cx_orbital"] = train_protonet(
                    ComplexAMC().to(DEV), smp_tr, a.budget, "orbital", unit=True, tag="cxOrb", log_every=1000)
                trunks["cx_euclid"] = train_protonet(
                    ComplexAMC().to(DEV), smp_tr, a.budget, "euclid", unit=True, tag="cxEuc", log_every=1000)
            # ---- 元信息 ----
            for nm, tr in trunks.items():
                meta.setdefault(nm, {"params": int(tr.n_params())})
            zprobe = zn[np.random.choice(len(zn), 256, replace=False)]
            for nm in ["cx_orbital", "cx_euclid", "real_euclid"]:
                if nm in trunks:
                    meta[nm]["equiv_delta"] = equivariance_error(trunks[nm], zprobe, device=DEV)
            # ---- 评估 ----
            for st in sigmas:
                smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                        q_per_class=15, seed=7100 + si + int(97 * st),
                                        snr_min=6, snr_max=18)
                inj = None if st == 0 else ("global", st)
                n = a.epi_eval
                Zs, Zq, yq = smp_te.sample(n, inject=inj)
                cell = {"split": si, "seed": seed, "sigma": float(st), "budget": a.budget,
                        "n_eval": n, "acc": {}}
                # identity（零训练参照）：原始 IQ 同时跑两套（raw=能量归一化，unit=再单位化）
                h_raw, k_raw = heads_on(Zs.astype(np.complex128), Zq.astype(np.complex128),
                                        yq, st, unit=False)
                for k2, v in h_raw.items():
                    cell["acc"][f"ident_{k2}"] = v.tolist()
                cell["kappa_ident"] = k_raw
                for nm, tr in trunks.items():
                    es = embed(tr, Zs).astype(np.complex128)
                    eq = embed(tr, Zq).astype(np.complex128)
                    h, kap = heads_on(es, eq, yq, st, unit=True)
                    for k2, v in h.items():
                        cell["acc"][f"{nm}_{k2}"] = v.tolist()
                    cell[f"kappa_{nm}"] = kap
                    cell["acc"][f"{nm}_euclid_raw"] = acc_of(
                        E.cls_euclid(np.ascontiguousarray(eq, dtype=np.complex128),
                                     E.proto_euclid(np.ascontiguousarray(es, dtype=np.complex128))), yq).tolist()
                recs.append(cell)
                summ = {k2: 100 * float(np.mean(v)) for k2, v in cell["acc"].items()}
                top = " ".join(f"{k2}={v:.1f}" for k2, v in summ.items()
                               if k2.endswith(("euclid", "orbital", "phML")))
                print(f"  [s{si}sd{seed} σ={st:.2f}] {top}", flush=True)
                print(f"      κ̂: ident={k_raw:.2f} " +
                      " ".join(f"{nm}={cell[f'kappa_{nm}']:.2f}" for nm in trunks), flush=True)
    json.dump({"meta": meta, "records": recs, "args": vars(a)},
              open(os.path.join(a.out, "logs", f"B17_{a.stage}{a.tag}.json"), "w"))
    print(f"B17[{a.stage}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
