"""B10：SOTA少样本AMC基线同协议对比（服务器3090, torch）。

被审稿人C点名的对比：与少样本AMC主流方法在同协议（相同测试episode、相同划分、相同SNR分层）下
的数字对比。基线选择（代表性而非穷举，正文声明）：
  SOTA-1  ProtoNet-CNN（端到端元训练，AMC少样本文献的事实标准骨架：4块1D-CNN）
  SOTA-2  ProtoNet-CNN + 随机相位旋转增广（增广变体）
本文方法（同episode评估）：
  Ours-1  identity特征 + PhaseMAP-ML（零训练，B8协议）
  Ours-2  ComplexTrunk端到端 + orbital目标（B7最优配置，预算3200）
参照：identity + euclid / orbital。
协议：6/2/3类划分×5（元训练只用train类；测试类episode 3-way 5-shot）；
      SNR分层{0,6,12,18} × 注入{none, pi/6, pi/2}；每格400 episodes；配对评估。
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


class AmcCNN(nn.Module):
    """AMC少样本文献标准骨架（非等变——这正是文献做法）。输入(2,L)实通道。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, z):                       # (B,L) complex
        x = torch.stack([z.real, z.imag], dim=1)
        return self.net(x).squeeze(-1)          # (B,128)


class AmcCLDNN(nn.Module):
    """CLDNN（Conv1D → LSTM → FC）：AMC 标杆混合架构（Ramjee et al. 2019
    arXiv:1901.05850 比较骨架；原始 CLDNN 概念属 O'Shea 等的 CNN+LSTM+DNN）。
    与 AmcCNN 同接口 (B,L) complex → (B,128)，引入时序递归家族差异。"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 128)

    def forward(self, z):                       # (B,L) complex
        x = torch.stack([z.real, z.imag], dim=1)
        h = self.conv(x)                        # (B,64,L/4)
        out, _ = self.lstm(h.transpose(1, 2))   # (B,L/4,64)
        return self.fc(out[:, -1, :])           # (B,128)


def proto_train(trunk, sampler, n_episodes, mode, device=DEV, bs=8, lr=1e-3):
    """端到端原型训练。mode: plain | aug(随机相位旋转)。"""
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    for it in range(n_episodes):
        Zs, Zq, yq = sampler.sample(bs, inject=None)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        if mode == "aug":
            th = torch.exp(1j * 2 * np.pi * torch.rand(B * N * k, 1, device=device))
            Zs_t = Zs_t.reshape(-1, L) * th
            thq = torch.exp(1j * 2 * np.pi * torch.rand(B * m, 1, device=device))
            Zq_t = Zq_t.reshape(-1, L) * thq
        es = trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1)
        eq = trunk(Zq_t.reshape(-1, L)).reshape(B, m, es.shape[-1])
        mu = es.mean(2)
        d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)) ** 2).sum(-1)
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 1000 == 0:
            print(f"    train {it+1}/{n_episodes} loss={loss.item():.4f}", flush=True)
    trunk.eval()
    return trunk


def embed(trunk, Z, device=DEV, bs=512):
    shape = Z.shape
    flat = Z.reshape(-1, shape[-1])
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            zt = torch.as_tensor(flat[i:i + bs], dtype=torch.complex64, device=device)
            outs.append(trunk(zt).cpu().numpy())
    return np.concatenate(outs, 0).reshape(shape[:-1] + (outs[0].shape[-1],))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--epi-eval", type=int, default=400)
    ap.add_argument("--epi-train", type=int, default=4000)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budgets", default=None)
    a = ap.parse_args()
    if a.quick:
        a.epi_eval, a.epi_train, a.splits = 100, 500, 2
    import sys as _s
    if 'budget' in _s.argv[0] or __import__('os').environ.get('B10_BUDGETS'):
        a.budgets = __import__('os').environ.get('B10_BUDGETS', a.budgets)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=a.splits)
    snr_levels = [0, 6, 12, 18]
    inj_grid = [("none", 0.0), ("global", np.pi / 6), ("global", np.pi / 2)]
    t0 = time.time()
    results = []
    for si, sp in enumerate(splits):
        print(f"=== split {si} ===", flush=True)
        # 训练两个SOTA变体（只用train类）
        smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                q_per_class=5, seed=4000 + si)
        budgets = [int(b) for b in a.budgets.split(",")] if a.budgets else [a.epi_train]
        cnns = {}
        for b in budgets:
            cnns[b] = {"plain": proto_train(AmcCNN().to(DEV), smp_tr, b, "plain"),
                       "aug": proto_train(AmcCNN().to(DEV), smp_tr, b, "aug"),
                       "cldnn": proto_train(AmcCLDNN().to(DEV), smp_tr, b, "plain"),
                       "cldnn_aug": proto_train(AmcCLDNN().to(DEV), smp_tr, b, "aug")}
            print(f"  trained budget={b} (cnn/cnn-aug/cldnn/cldnn-aug)", flush=True)
        for sl in snr_levels:
            for mode, strength in inj_grid:
                smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                        q_per_class=15, seed=5000 + 7 * si + sl,
                                        snr_min=sl, snr_max=sl)
                inj = None if mode == "none" else (mode, strength)
                n = a.epi_eval
                Zs, Zq, yq = smp_te.sample(n, inject=inj)
                Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
                acc = {}
                # SOTA-1/2：端到端ProtoNet-CNN（3-way读出）
                for b, nets in cnns.items():
                    for tag, net in [("sota_protonet_cnn", nets["plain"]),
                                     ("sota_protonet_cnn_aug", nets["aug"]),
                                     ("sota_protonet_cldnn", nets["cldnn"]),
                                     ("sota_protonet_cldnn_aug", nets["cldnn_aug"])]:
                        name = f"{tag}_b{b}" if len(budgets) > 1 else tag
                        es = embed(net, Zs).astype(np.float64)      # (B,N,k,C)
                        eq = embed(net, Zq).astype(np.float64)      # (B,m,C)
                        d2 = ((eq[:, :, None, :] - es.mean(2)[:, None, :, :]) ** 2).sum(-1)
                        acc[name] = float((d2.argmin(-1) == yq).mean())
                # Ours-1：identity + PhaseMAP-ML（零训练）
                for e in range(n):
                    pass
                acc_ours = np.zeros(n)
                # 分块算（内存）
                for g0 in range(0, n, 100):
                    sl_ = slice(g0, min(g0 + 100, n))
                    mu_pm, aux = E.phasemap_em(Zs128[sl_])
                    pred = E.cls_marginal(Zq128[sl_], mu_pm, aux)
                    acc_ours[sl_] = (pred == yq[sl_]).mean(1)
                acc["ours_phasemap_ml_identity"] = float(acc_ours.mean())
                # 参照：identity euclid/orbital
                acc["ref_identity_euclid"] = float(
                    (E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean(1).mean())
                acc["ref_identity_orbital"] = float(
                    (E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean(1).mean())
                rec = {"split": si, "snr": sl, "inject_mode": mode,
                       "inject_strength": float(strength), "acc": acc,
                       "n_eval": n, "n_way": 3, "k_shot": 5}
                results.append(rec)
                print(f"  [split{si} SNR{sl} {mode}{strength:.2f}] " +
                      " ".join(f"{k2}={100*v:.1f}" for k2, v in acc.items()), flush=True)
    json.dump(results, open(os.path.join(a.out, "logs", "B10_sota_compare.json"), "w"), indent=1)
    print(f"B10 DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
