"""B6/B7 服务器入口（torch, RTX 3090, conda env msb）。

B6: 等变诊断 + 冻结编码器少样本（随机冻结 / 监督预训练冻结 / 实值对照）
B7: 端到端原型训练（euclid-proto / 增广-proto / 轨道-proto）× 元训练预算 × 3种子
    -> 预注册发现: 增广的商不变性需要更多元训练数据（少样本下失效）

用法: python run_b67.py --mode b6   /   --mode b7   /  --mode all
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe import estim as E
from cvfe.episodes import EpisodeSampler
from cvfe.torch_nets import ComplexTrunk, RealTrunk, equivariance_error

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHODS_EMB = ["euclid", "orbital", "tta_euclid", "phasemap", "phasemap_ml", "phasemap_w"]


def embed(trunk, Z, device=DEV, bs=512):
    """Z:(E,N,k,L) or (E,m,L) complex -> embeddings, 保持形状。"""
    shape = Z.shape
    flat = Z.reshape(-1, shape[-1])
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            zt = torch.as_tensor(flat[i:i + bs], dtype=torch.complex64, device=device)
            e = trunk(zt)
            outs.append(e.cpu().numpy())
    emb = np.concatenate(outs, 0).reshape(shape[:-1] + (outs[0].shape[-1],))
    return emb


def pretrain_supervised(trunk, Z, y, epochs=12, lr=1e-3, bs=256, device=DEV, wd=1e-4):
    """监督预训练：线性头 + CE。Z:(n,L) complex。返回训练后 trunk。"""
    n_cls = int(y.max()) + 1
    with torch.no_grad():
        e0 = trunk(torch.as_tensor(Z[:2], device=device))
        dim = 2 * e0.shape[-1] if e0.is_complex() or torch.is_complex(e0) else e0.shape[-1]
    head = nn.Linear(dim, n_cls).to(device)
    opt = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=wd)
    ds = torch.as_tensor(Z, device=device)
    ys = torch.as_tensor(y, device=device)
    n = len(Z)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, corr = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            emb = trunk(ds[idx])
            if torch.is_complex(emb):
                emb = torch.cat([emb.real, emb.imag], dim=-1)
            logits = head(emb)
            loss = F.cross_entropy(logits, ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
            corr += (logits.argmax(1) == ys[idx]).sum().item()
        print(f"  pretrain ep{ep}: loss={tot/n:.4f} acc={corr/n:.3f}", flush=True)
    return trunk


def eval_suite_on_emb(emb_s, emb_q, yq, methods, kappa_meta=np.pi / 3):
    """冻结嵌入上的估计层评估。B6 表的 phasemap_w/phasemap_unc 曾与 phasemap
    逐位相同（else 分支未传 hetero_weights / uncertainty，2026-10-02 审计 S5），
    现按 run_method 注册表语义分发。"""
    acc = {}
    for m in methods:
        if m == "euclid":
            pred = E.cls_euclid(emb_q, E.proto_euclid(emb_s))
        elif m == "orbital":
            pred = E.cls_orbital(emb_q, E.proto_orbital(emb_s))
        elif m == "tta_euclid":
            pred = E.cls_orbital(emb_q, E.proto_euclid(emb_s))
        elif m == "phasemap":
            mu, aux = E.phasemap_em(emb_s, kappa_meta=kappa_meta)
            pred = E.cls_orbital(emb_q, mu, aux)
        elif m == "phasemap_ml":
            mu, aux = E.phasemap_em(emb_s, kappa_meta=kappa_meta)
            pred = E.cls_marginal(emb_q, mu, aux)
        elif m == "phasemap_w":                       # 异方差精度加权（P3 组件）
            mu, aux = E.phasemap_em(emb_s, kappa_meta=kappa_meta, hetero_weights=True)
            pred = E.cls_orbital(emb_q, mu, aux)
        elif m == "phasemap_unc":                     # 一阶不确定性修正
            mu, aux = E.phasemap_em(emb_s, kappa_meta=kappa_meta)
            pred = E.cls_orbital(emb_q, mu, aux, uncertainty=True)
        else:
            raise ValueError(m)
        acc[m] = float((pred == yq).mean())
    return acc


def run_b6(zn, y, snr, out, epi=400):
    """等变诊断 + 冻结编码器少样本。"""
    res = {"equivariance": {}, "frozen_fewshot": []}
    ctrunk = ComplexTrunk().to(DEV)
    rtrunk = RealTrunk().to(DEV)
    print(f"complex trunk real params = {ctrunk.n_real_params()}", flush=True)
    print(f"real trunk real params   = {rtrunk.n_real_params()}", flush=True)
    res["params"] = {"complex": ctrunk.n_real_params(), "real": rtrunk.n_real_params()}

    idx = np.random.RandomState(5).choice(len(zn), 512, replace=False)
    res["equivariance"]["complex_trunk"] = equivariance_error(ctrunk, zn[idx], device=DEV)
    res["equivariance"]["real_trunk"] = equivariance_error(rtrunk, zn[idx], device=DEV)
    print(f"equivariance delta: complex={res['equivariance']['complex_trunk']:.4f} "
          f"real={res['equivariance']['real_trunk']:.4f}", flush=True)

    # 冻结编码器少样本：随机冻结复trunk（11类5-way）；监督预训练冻结（6训练类->3测试类3-way）
    from cvfe.episodes import make_class_splits
    splits = make_class_splits(n_splits=5)
    for inj_name, inj in [("none", None), ("global_pi/3", ("global", np.pi / 3)),
                          ("global_pi", ("global", np.pi))]:
        # (a) 随机冻结复trunk, 11类 5-way
        smp = EpisodeSampler(zn, y, snr, classes=list(range(11)), n_way=5, k_shot=5,
                             q_per_class=15, seed=123)
        Zs, Zq, yq = smp.sample(epi, inject=inj)
        es, eq = embed(ctrunk, Zs), embed(ctrunk, Zq)
        acc = eval_suite_on_emb(es.astype(np.complex128), eq.astype(np.complex128), yq,
                                METHODS_EMB)
        res["frozen_fewshot"].append({"setting": "complex_random_11class_5way",
                                      "inject": inj_name, "acc": acc})
        print(f"[b6 {inj_name}] complex_random: " +
              " ".join(f"{k}={100*v:.1f}" for k, v in acc.items()), flush=True)
        # (b) 监督预训练冻结（每个 split 一个）, 测试类 3-way
        for si, sp in enumerate(splits[:3]):
            tr = np.where(np.isin(y, sp["train"]))[0]
            ctr = pretrain_supervised(ComplexTrunk().to(DEV), zn[tr], y[tr], epochs=8)
            smp_t = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                   q_per_class=15, seed=321 + si)
            Zs3, Zq3, yq3 = smp_t.sample(epi, inject=inj)
            es3, eq3 = embed(ctr, Zs3), embed(ctr, Zq3)
            acc3 = eval_suite_on_emb(es3.astype(np.complex128), eq3.astype(np.complex128),
                                     yq3, METHODS_EMB)
            rtr = pretrain_supervised(RealTrunk().to(DEV), zn[tr], y[tr], epochs=8)
            ers, eqr = embed(rtr, Zs3), embed(rtr, Zq3)
            acc_r = {"euclid": float((E.cls_euclid(eqr, E.proto_euclid(ers)) == yq3).mean()),
                     "whiten": float((E.cls_mahalanobis_diag(
                         eqr, E.proto_euclid(ers), E.whiten_aux(eqr)) == yq3).mean())}
            res["frozen_fewshot"].append({"setting": f"pretrained_split{si}_3way",
                                          "inject": inj_name, "acc_complex": acc3,
                                          "acc_real": acc_r})
            print(f"[b6 {inj_name} split{si}] complex: " +
                  " ".join(f"{k}={100*v:.1f}" for k, v in acc3.items()) +
                  " | real: " + " ".join(f"{k}={100*v:.1f}" for k, v in acc_r.items()),
                  flush=True)
    json.dump(res, open(os.path.join(out, "logs", "B6_encoder.json"), "w"), indent=1)


def _epi_batch(sampler, n, inject, device):
    Zs, Zq, yq = sampler.sample(n, inject=inject)
    return (torch.as_tensor(Zs, dtype=torch.complex64, device=device),
            torch.as_tensor(Zq, dtype=torch.complex64, device=device),
            torch.as_tensor(yq, device=device))


def proto_loss(trunk, Zs_e, Zq_e, yq_e, mode, n_way, device):
    """在 trunk 复嵌入空间计算原型损失。
    mode: euclid | aug(随机相位旋转增广) | orbital（嵌入上轨道距离）"""
    B, N, k, L = Zs_e.shape
    m = Zq_e.shape[1]
    if mode == "aug":
        th = torch.exp(1j * 2 * np.pi * torch.rand(B * N * k, 1, device=device))
        Zs_e = Zs_e.reshape(-1, L) * th
        thq = torch.exp(1j * 2 * np.pi * torch.rand(B * m, 1, device=device))
        Zq_e = Zq_e.reshape(-1, L) * thq
    es = trunk(Zs_e.reshape(-1, L)).reshape(B, N, k, -1)       # (B,N,k,C) complex
    eq = trunk(Zq_e.reshape(-1, L)).reshape(B, m, -1)
    mu = es.mean(2)                                            # (B,N,C)
    if mode == "orbital":
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = (eq.abs() ** 2).sum(-1).unsqueeze(-1) + (mu.abs() ** 2).sum(-1).unsqueeze(1)             - 2 * ip.abs()
    else:
        d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)) ** 2).abs().sum(-1)   # (B,m,N)
    return F.cross_entropy(-d2.reshape(-1, n_way), yq_e.reshape(-1))

def run_b7(zn, y, snr, out, seeds=(0, 1, 2), budgets=(200, 800, 3200),
           train_sigmas=(0.0, np.pi / 3, np.pi)):
    """端到端：euclid-proto / aug-proto / orbital-proto；预算扫描。"""
    from cvfe.episodes import make_class_splits
    splits = make_class_splits(n_splits=5)
    sp = splits[0]
    res = []
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=555)
    for budget in budgets:
        for mode in ("euclid", "aug", "orbital"):
            for seed in seeds:
                torch.manual_seed(seed)
                trunk = ComplexTrunk().to(DEV)   # 等变复trunk（嵌入为复向量）
                opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
                for it in range(budget):
                    sg = train_sigmas[it % len(train_sigmas)]
                    Zs_t, Zq_t, yq_t = _epi_batch(smp_tr, 4, ("global", float(sg)), DEV)
                    Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    loss = proto_loss(trunk, Zs_t, Zq_t, yq_t, mode, 5, DEV)
                    opt.zero_grad(); loss.backward(); opt.step()
                # 评估：测试类 3-way, 注入 σ_θ 网格
                smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                        q_per_class=15, seed=999 + seed)
                evals = {}
                for ev_sig in (0.0, np.pi / 3, np.pi):
                    Zs_e, Zq_e, yq_e = smp_te.sample(200, inject=("global", float(ev_sig)))
                    Zs_t = torch.as_tensor(Zs_e, dtype=torch.complex64, device=DEV)
                    Zq_t = torch.as_tensor(Zq_e, dtype=torch.complex64, device=DEV)
                    Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                    with torch.no_grad():
                        Bz, Nz, kz, Lz = Zs_t.shape
                        es = trunk(Zs_t.reshape(-1, Lz)).reshape(Bz, Nz, kz, -1).mean(2)
                        eq = trunk(Zq_t.reshape(-1, Lz)).reshape(Bz, -1, Zs_t.shape[-1] * 0 + es.shape[-1])
                    es_np = es.cpu().numpy().astype(np.complex128)
                    eq_np = eq.cpu().numpy().astype(np.complex128)
                    d2 = ((np.abs(eq_np[:, :, None, :] - es_np[:, None, :, :]) ** 2)).sum(-1)
                    acc = float((d2.argmin(-1) == yq_e).mean())
                    evals[f"sigma_{ev_sig:.2f}"] = acc
                res.append({"budget": budget, "mode": mode, "seed": seed, "acc": evals})
                print(f"[b7] budget={budget} mode={mode} seed={seed}: " +
                      " ".join(f"{k}={100*v:.1f}" for k, v in evals.items()), flush=True)
    json.dump(res, open(os.path.join(out, "logs", "B7_endtoend.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--epi", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    t0 = time.time()
    if a.mode in ("b6", "all"):
        run_b6(zn, y, snr, a.out, epi=a.epi)
    if a.mode in ("b7", "all"):
        run_b7(zn, y, snr, a.out)
    print(f"B6/B7 DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
