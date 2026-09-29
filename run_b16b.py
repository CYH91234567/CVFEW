"""B16b：端到端 κ 头（等变复trunk + 可微κ预测 + 可微软对齐EM + 边际似然分类，全torch）。

设计（针对B16a负结果与B14边界定位）：
  - κ̂ 由学习到的嵌入上的对齐统计量预测（trunk可学习"暴露相位宽度"的特征）；
  - κ̂ 馈入软对齐E步（γ网格比）与边际似然分类（B14：原型阶段必须用到κ）；
  - 训练episode的 σ_θ ~ {0,π/6,π/3,π/2,2π/3,π} 均匀 × fade~{0,0.5}（迫使解耦，
    标签只含σ_θ；fade作为纯干扰因子）；
  - 纯任务损失（CE），κ̂无直接监督——校准评估揭示它是被学出还是被trunk旁路。

评估（预注册）：
  SC1 校准：测试类episode上 κ̂ vs 真σ_θ 的Spearman > 0.8（B16a为≈0）
  SC2 失效格（fade+π/6）：joint ≥ 固定κ=π/3 + 1pt
  SC3 安全（σ_θ=π）：joint 与 oracle-κ 差 ≤ 1pt
基线（同episode）：identity{euclid, orbital, phML-profile, phML-oracle} +
                  trunk{+euclid, +fixed-κ, +κ̂, +oracle-κ}
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
from cvfe.torch_nets import ComplexTrunk

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIGMA_GRID = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
HELD_OUT = [np.pi / 4, 3 * np.pi / 4]
NGRID = 64
TH = torch.linspace(-np.pi, np.pi, NGRID, device=DEV)


class KappaHead(nn.Module):
    """对齐统计量 -> κ ∈ (1e-3, π]。输入 [R_pooled, log_align]。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())

    def forward(self, feats):
        return self.net(feats).clamp(1e-3, float(np.pi)).squeeze(-1)


def wrapped_prior_torch(kappa, n_img=4):
    """pr(θ_g; κ)：可微缠绕高斯。kappa:(B,) -> (B,NGRID)"""
    imgs = TH[:, None] + 2 * np.pi * torch.arange(-n_img, n_img + 1,
                                                  device=kappa.device)[None, :]   # (G,J)
    pr = torch.exp(-0.5 * (imgs[None] / kappa[:, None, None]) ** 2).sum(-1)         # (B,G)
    return pr / pr.sum(-1, keepdim=True).clamp_min(1e-12)


def gamma_grid_torch(rho, psi, kappa):
    """γ = E[e^{-iθ}|z] 的网格比。rho,psi:(B,K,k) kappa:(B,)。"""
    pr = wrapped_prior_torch(kappa)[:, None, None, :]           # (B,1,1,G)
    u = psi[..., None] - TH                                      # (B,K,k,G)
    w = torch.exp(rho[..., None] * (torch.cos(u) - 1.0)) * pr    # 平移防溢出
    den = w.sum(-1).clamp_min(1e-30)
    num = (w * torch.exp(-1j * TH)).sum(-1)
    return num / den


def unit_norm(e):
    return e / e.abs().norm(dim=-1, keepdim=True).clamp_min(1e-8)


def align_iter(es, eq, kappa, n_iter=3):
    """软对齐EM（给定κ）。es:(B,N,k,C) eq:(B,m,C) 均已单位化。
    返回 mu:(B,N,C), sigma2:(B,N), gamma:(B,N,k)。"""
    B, N, k, C = es.shape
    mu = es.mean(2)
    gam = None
    for _ in range(n_iter):
        ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
        psi = torch.angle(ip)
        rho = 2.0 * ip.abs() / C                                  # 单位化嵌入的工作尺度
        rho = rho / rho.detach().median().clamp_min(1e-6)         # 全局尺度稳定（工作点选择）
        gam = gamma_grid_torch(rho, psi, kappa)              # kappa:(B,) -> 内部广播
        mu = (gam[..., None] * es).sum(2) / k
        res2 = ((es - mu[:, :, None, :]).abs() ** 2).sum(-1)       # 对齐后残差
        sigma2 = res2.mean(dim=2) / C + 1e-6                       # (B,N) 逐类
    return mu, sigma2, gam


def joint_forward(trunk, khead, Zs, Zq):
    """前向：嵌入 -> 初始对齐(κ=π/2中性) -> 统计 -> κ̂ -> 终对齐 -> 分类logits。
    返回 logits:(B,m,N), kappa_hat:(B,), aux。"""
    B, N, k, L = Zs.shape
    m = Zq.shape[1]
    es = unit_norm(trunk(Zs.reshape(-1, L)).reshape(B, N, k, -1))
    eq = unit_norm(trunk(Zq.reshape(-1, L)).reshape(B, m, -1))
    kappa0 = torch.full((B,), np.pi / 2, device=Zs.device)
    mu0, s20, gam0 = align_iter(es, eq, kappa0, n_iter=2)
    # 对齐统计量（可微）
    ip0 = torch.einsum("bnc,bnkc->bnk", mu0.conj(), es)
    R = torch.exp(1j * torch.angle(ip0)).mean(dim=(1, 2)).abs()      # (B,) 池化合向量
    log_al = torch.log(ip0.abs().mean(dim=(1, 2)).clamp_min(1e-8))
    kappa_hat = khead(torch.stack([R, log_al], dim=1))
    mu, sigma2, gam = align_iter(es, eq, kappa_hat, n_iter=3)
    # 边际似然分类（可微，含σ̂²_c与κ̂）
    ip_q = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
    psi_q = torch.angle(ip_q)
    rho_q = 2.0 * ip_q.abs() / sigma2[:, None, :].clamp_min(1e-8)
    pr = wrapped_prior_torch(kappa_hat)[:, None, None, :]
    w = torch.exp(rho_q[..., None] * (torch.cos(psi_q[..., None] - TH) - 1.0)) * pr
    c0 = w.sum(-1).clamp_min(1e-30)
    p_dim = eq.shape[-1]
    score = (-p_dim * torch.log(sigma2.clamp_min(1e-8))[:, None, :]
             - ((eq.abs() ** 2).sum(-1, keepdim=True) + (mu.abs() ** 2).sum(-1)[:, None, :])
             / sigma2[:, None, :].clamp_min(1e-8)
             + rho_q + torch.log(c0))
    return score, kappa_hat, {"mu": mu, "sigma2": sigma2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--epi-train", type=int, default=4000)
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epi_train, a.epi_eval, a.splits, a.seeds = 600, 80, 2, 1
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=a.splits)
    t0 = time.time()
    all_res = []
    for si, sp in enumerate(splits):
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=6100 + si, snr_min=6, snr_max=12)
        for seed in range(a.seeds):
            torch.manual_seed(100 * si + seed)
            trunk = ComplexTrunk().to(DEV)
            khead = KappaHead().to(DEV)
            opt = torch.optim.AdamW(list(trunk.parameters()) + list(khead.parameters()),
                                    lr=1e-3, weight_decay=1e-4)
            for it in range(a.epi_train):
                st_e = float(SIGMA_GRID[it % len(SIGMA_GRID)])
                fq_e = float(FADE := [0.0, 0.5][(it // len(SIGMA_GRID)) % 2])
                Zs, Zq, yq = smp_tr.sample(4, inject=(None if st_e == 0 else ("global", st_e)))
                Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
                Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
                Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                if fq_e > 0:                                   # fade作为纯干扰因子
                    h = torch.where(torch.rand(Zs_t.shape[:-1], device=DEV) < fq_e,
                                    torch.full(Zs_t.shape[:-1], 0.1, device=DEV),
                                    torch.ones(Zs_t.shape[:-1], device=DEV))
                    Zs_t = Zs_t * h.unsqueeze(-1)
                score, khat, _ = joint_forward(trunk, khead, Zs_t, Zq_t)
                loss = F.cross_entropy(score.reshape(-1, score.shape[-1]),
                                       torch.as_tensor(yq.reshape(-1), device=DEV))
                opt.zero_grad(); loss.backward(); opt.step()
                if (it + 1) % 1000 == 0:
                    print(f"  [s{si} seed{seed}] ep{it+1} loss={loss.item():.4f} "
                          f"khat={float(khat.mean()):.3f}", flush=True)
            # ---- 评估 ----
            trunk.eval(); khead.eval()
            for st_e in SIGMA_GRID + HELD_OUT:
                for fq_e in [0.0, 0.5]:
                    smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                            q_per_class=15, seed=7100 + si + int(97 * st_e) + int(11 * fq_e),
                                            snr_min=6, snr_max=12)
                    inj = None if st_e == 0 else ("global", st_e)
                    Zs, Zq, yq = smp_te.sample(a.epi_eval, inject=inj)
                    Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
                    acc = {}
                    acc["id_euclid"] = float((E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean())
                    acc["id_orbital"] = float((E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean())
                    mu_p, aux_p = E.phasemap_em(Zs128)
                    acc["id_phML_profile"] = float((E.cls_marginal(Zq128, mu_p, aux_p) == yq).mean())
                    mu_o, aux_o = E.phasemap_em(Zs128, sigma_th=(st_e if st_e > 0 else 1e-3))
                    acc["id_phML_oracle"] = float((E.cls_marginal(Zq128, mu_o, aux_o) == yq).mean())
                    # trunk 系（分块）
                    n = a.epi_eval
                    acc_t = {k2: np.zeros(n) for k2 in
                             ["tr_euclid", "tr_fix_pi3", "tr_joint", "tr_oracle"]}
                    with torch.no_grad():
                        for g0 in range(0, n, 50):
                            sl_ = slice(g0, min(g0 + 50, n))
                            Zs_t = torch.as_tensor(Zs[sl_], dtype=torch.complex64, device=DEV)
                            Zq_t = torch.as_tensor(Zq[sl_], dtype=torch.complex64, device=DEV)
                            Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                            Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                            B, N, k, L = Zs_t.shape
                            esn = unit_norm(trunk(Zs_t.reshape(-1, L))).reshape(B, N, k, -1)
                            eqn = unit_norm(trunk(Zq_t.reshape(-1, L))).reshape(B, Zq_t.shape[1], -1)
                            # trunk euclid
                            mu_e = esn.mean(2)
                            d2e = ((eqn.unsqueeze(2) - mu_e.unsqueeze(1)).abs() ** 2).sum(-1)
                            acc_t["tr_euclid"][sl_] = (d2e.argmin(-1).cpu().numpy() == yq[sl_]).mean(1)
                            # trunk + 各κ（软对齐EM+边际分类）
                            for name, kap in [("tr_fix_pi3", np.pi / 3),
                                              ("tr_oracle", max(st_e, 1e-3))]:
                                mu_k, s2_k, _ = align_iter(esn, eqn,
                                    torch.full((B,), kap, device=DEV), n_iter=5)
                                ip_q = torch.einsum("bmc,bnc->bmn", eqn.conj(), mu_k)
                                rho_q = 2.0 * ip_q.abs() / s2_k[:, None, :].clamp_min(1e-8)
                                pr = wrapped_prior_torch(torch.full((B,), kap, device=DEV))[:, None, None, :]
                                wq = torch.exp(rho_q[..., None] * (torch.cos(torch.angle(ip_q)[..., None] - TH) - 1.0)) * pr
                                sc = (-esn.shape[-1] * torch.log(s2_k.clamp_min(1e-8))[:, None, :]
                                      - ((eqn.abs() ** 2).sum(-1, keepdim=True) + (mu_k.abs() ** 2).sum(-1)[:, None, :])
                                      / s2_k[:, None, :].clamp_min(1e-8) + rho_q + torch.log(wq.sum(-1).clamp_min(1e-30)))
                                acc_t[name][sl_] = (sc.argmax(-1).cpu().numpy() == yq[sl_]).mean(1)
                            sc_j, khat_j, _ = joint_forward(trunk, khead, Zs_t, Zq_t)
                            acc_t["tr_joint"][sl_] = (sc_j.argmax(-1).cpu().numpy() == yq[sl_]).mean(1)
                    for k2 in acc_t:
                        acc[k2] = float(acc_t[k2].mean())
                    khat_mean = None
                    with torch.no_grad():
                        # 记录κ̂校准（小批）
                        kh_list = []
                        for g0 in range(0, n, 50):
                            sl_ = slice(g0, min(g0 + 50, n))
                            Zs_t = torch.as_tensor(Zs[sl_], dtype=torch.complex64, device=DEV)
                            Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                            zq_n = torch.as_tensor(Zq[sl_], dtype=torch.complex64, device=DEV)
                            zq_n = zq_n / zq_n.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                            _, khat_j, _ = joint_forward(trunk, khead, Zs_t, zq_n)
                            kh_list.append(khat_j.cpu().numpy())
                        khat_mean = float(np.concatenate(kh_list).mean())
                    rec = {"split": si, "seed": seed, "sigma_th_true": float(st_e),
                           "fade": fq_e, "held_out": st_e in HELD_OUT,
                           "kappa_hat_mean": khat_mean, "acc": acc}
                    all_res.append(rec)
                    print(f"  [s{si}sd{seed}] st={st_e:.2f} fade={fq_e}: khat={khat_mean:.2f} "
                          f"euclid={100*acc['id_euclid']:.0f} orbital={100*acc['id_orbital']:.0f} "
                          f"joint={100*acc['tr_joint']:.0f} tr_fix={100*acc['tr_fix_pi3']:.0f} "
                          f"tr_oracle={100*acc['tr_oracle']:.0f}", flush=True)
    json.dump(all_res, open(os.path.join(a.out, "logs", "B16b_joint_kappa.json"), "w"), indent=1)
    print(f"B16b DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
