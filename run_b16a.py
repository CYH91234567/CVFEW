"""B16a：元学习κ回归器（meta-learned empirical-Bayes prior for sigma_theta）。

动机（B14结论）：episode级profile-kappa把幅度衰落混淆进相位宽度估计（fade+小sigma_theta劣势
的根因）。修复：在train类上生成 {sigma_theta 网格 × fade 网格} 双因子注入episode，训练回归器
从episode统计量预测sigma_theta（标签只含sigma_theta）——学会解耦两因子；测试类episode上
用 kappa_hat 作为 PhaseMAP 的固定先验。

特征（每episode，5维，全部来自phasemap_em的aux与支持集）：
  [log mean sigma2_c, R_aligned(池化对齐合向量), mean|gamma|, std|gamma|, log mean frame energy]
评估：
  1) kappa校准（测试类episode，sigma_theta网格+held-out {pi/4, 3pi/4} × fade{0,0.5}）
  2) 失效格(fade=0.5, sigma_theta=pi/6)上：profile-k vs kappa_hat 的 PhaseMAP-ML 准确率
     + 无先验/固定先验对照
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe import estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIGMA_GRID = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
HELD_OUT = [np.pi / 4, 3 * np.pi / 4]
FADE_GRID = [0.0, 0.5]


def episode_features(Zs):
    """Zs:(E,K,k,p) -> 特征 (E,5)。用phasemap_em的aux与原始支持集统计。"""
    Epi, K, k, p = Zs.shape
    mu, aux = E.phasemap_em(Zs)
    gam = aux["gam"]                                   # (E,K,k) 对齐强度
    ip = np.einsum("ekp,ekjp->ekj", mu.conj(), Zs)
    ph = np.angle(ip)
    R = np.abs(np.exp(1j * ph).mean(axis=(1, 2)))      # 池化对齐合向量
    feats = np.stack([
        np.log(np.maximum(aux["sigma2"].mean(1), 1e-12)).ravel(),
        R,
        np.abs(gam).mean(axis=(1, 2)).ravel(),
        np.abs(gam).std(axis=(1, 2)).ravel(),
        np.log(np.maximum(np.mean(np.abs(Zs) ** 2, axis=(1, 2, 3)), 1e-12)),
    ], axis=1)
    return feats.astype(np.float32), aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_train, a.n_eval, a.splits, a.epochs = 300, 100, 2, 150
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=a.splits)
    t0 = time.time()
    all_res = []
    for si, sp in enumerate(splits):
        print(f"=== split {si} ===", flush=True)
        # ---- 训练集：train类 × {sigma网格 × fade网格}（标签只含sigma_theta）----
        Xs, Ys = [], []
        per = max(1, a.n_train // (len(SIGMA_GRID) * len(FADE_GRID)))
        for st in SIGMA_GRID:
            for fq in FADE_GRID:
                smp = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                                     q_per_class=5, seed=6000 + 13 * si + int(100 * st) + int(10 * fq),
                                     snr_min=12, snr_max=18)
                inj = None if st == 0.0 else ("global", st)
                Zs, _, _ = smp.sample(per, inject=inj)
                f, _ = episode_features(Zs.astype(np.complex128))
                Xs.append(f); Ys.append(np.full(per, st, dtype=np.float32))
        Xtr = np.concatenate(Xs); Ytr = np.concatenate(Ys)
        # ---- 回归器（torch 小MLP）----
        torch.manual_seed(si)
        net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
                            LambdaClamp(1e-3, np.pi)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        Xtr_t = torch.as_tensor(Xtr, device=DEV)
        Ytr_t = torch.as_tensor(Ytr, device=DEV)
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=DEV)
            for i in range(0, len(Xtr), 64):
                idx = perm[i:i + 64]
                pred = net(Xtr_t[idx]).squeeze(-1)
                loss = F_mse(pred, Ytr_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"  regressor trained; train MAE={float((net(Xtr_t).squeeze(-1)-Ytr_t).abs().mean()):.4f}",
              flush=True)
        # ---- 评估1：测试类episode的kappa校准（含held-out sigma）----
        calib = []
        for st in SIGMA_GRID + HELD_OUT:
            for fq in FADE_GRID:
                smp = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                     q_per_class=15, seed=7000 + 17 * si + int(100 * st) + int(10 * fq),
                                     snr_min=6, snr_max=12)
                inj = None if st == 0.0 else ("global", st)
                Zs, _, _ = smp.sample(a.n_eval, inject=inj)
                f, _ = episode_features(Zs.astype(np.complex128))
                with torch.no_grad():
                    kh = net(torch.as_tensor(f, device=DEV)).squeeze(-1).cpu().numpy()
                calib.append({"sigma_th_true": float(st), "fade": fq,
                              "kappa_hat_mean": float(kh.mean()),
                              "kappa_hat_std": float(kh.std()),
                              "held_out": st in HELD_OUT})
                print(f"  [calib] true={st:.2f} fade={fq}: khat={kh.mean():.3f}±{kh.std():.3f}"
                      + ("  (held-out)" if st in HELD_OUT else ""), flush=True)
        # ---- 评估2：失效格 fade=0.5, sigma_theta=pi/6 的准确率对比 ----
        smp_fail = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                                  q_per_class=15, seed=8000 + si, snr_min=6, snr_max=12)
        Zs, Zq, yq = smp_fail.sample(a.n_eval, inject=("global", np.pi / 6))
        Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
        f_fail, aux_fail = episode_features(Zs128)
        with torch.no_grad():
            kh_fail = net(torch.as_tensor(f_fail, device=DEV)).squeeze(-1).cpu().numpy()
        accs = {}
        accs["euclid"] = float((E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean(1).mean())
        accs["orbital"] = float((E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean(1).mean())
        mu_pm, aux_pm = E.phasemap_em(Zs128)          # profile-k（无先验）
        accs["phML_profile"] = float((E.cls_marginal(Zq128, mu_pm, aux_pm) == yq).mean(1).mean())
        # 逐episode固定kappa=khat（按khat分桶复用计算）
        acc_meta = eval_with_per_episode_kappa(Zs128, Zq128, yq, kh_fail)
        accs["phML_meta_kappa"] = float(acc_meta.mean())
        mu_o, aux_o = E.phasemap_em(Zs128, sigma_th=np.pi / 6)   # oracle kappa（真值）
        accs["phML_oracle_kappa"] = float((E.cls_marginal(Zq128, mu_o, aux_o) == yq).mean(1).mean())
        print(f"  [fail-cell] {json.dumps({k2: round(100*v,1) for k2, v in accs.items()})}", flush=True)
        all_res.append({"split": si, "calibration": calib, "fail_cell_acc": accs,
                        "fail_cell_kappa_hat_mean": float(kh_fail.mean()),
                        "fail_cell_kappa_hat_std": float(kh_fail.std())})
    json.dump(all_res, open(os.path.join(a.out, "logs", "B16_meta_kappa.json"), "w"), indent=1)
    print(f"B16a DONE in {(time.time()-t0)/60:.1f} min")


def F_mse(pred, tgt):
    return torch.mean((pred - tgt) ** 2)


class LambdaClamp(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return x.clamp(self.lo, self.hi)


def eval_with_per_episode_kappa(Zs128, Zq128, yq, kh):
    """逐episode以khat为固定先验的PhaseMAP-ML准确率（按khat分桶复用计算）。"""
    n = len(kh)
    acc = np.zeros(n)
    order = np.argsort(kh)
    bins = np.array_split(order, 10)                     # 10个kappa桶
    for b in bins:
        if len(b) == 0:
            continue
        khat = float(np.median(kh[b]))
        mu_b, aux_b = E.phasemap_em(Zs128[b], sigma_th=float(khat))
        pred = E.cls_marginal(Zq128[b], mu_b, aux_b)
        acc[b] = (pred == yq[b]).mean(1)
    return acc


if __name__ == "__main__":
    main()
