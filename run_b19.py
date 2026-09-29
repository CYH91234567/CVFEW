"""B19：等变复 CNN 训练失败的诊断性修复（服务器 3090；PREREG_B19）。

臂（ComplexAMC2，nets_b19）：
  v0_mean   原样（均值池化，随机初始化）——复现 B17 失败对照
  v1_hann   Hann 平滑初始化 + 均值池化（严格等变复输出）——主修复臂
  v13_gated Hann 初始化 + 幅度门控池化（严格等变复输出）
  v2_power  功率池化（不变实嵌入，折中臂）
  real      RealAMC 实对照（B17 已知 ~91.4@split0）
目标 × {euclid, orbital}（v2_power 仅 euclid）。评估：冻结嵌入 + euclid/orbital/phML 头，
σ_θ 注入 {0, π/3, π}，等变 δ，末层相消比，嵌入 Fisher 比。
synth 模式：σ_θ=0 合成（p=128, ρ=0.3, r=1）上的 v0 vs v1——D4 纯优化对照。
smoke 模式：budget=30 全管线冒烟。
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, synth as S
from cvfe import estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import ComplexAMC, RealAMC, unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, fisher_ratio

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def l1pca_proto(es, n_iter=5):
    """可微 L1-PCA 原型（= estim.orbital_align 的 torch 版，单重启从均值出发）。

    mu_{t+1} = (1/k) Σ_j z_j · e^{-i arg(mu_t^H z_j)}，对齐相位 stop-gradient
    （EM 式坐标上升：每步把对齐权重视为常数，梯度只经 z_j 流动——与估计层
    proto_orbital 严格同构，且规避 angle 在 0 处的梯度爆炸）。es:(B,N,k,C)。
    """
    mu = es.mean(2)
    for _ in range(n_iter):
        ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
        w = torch.exp(-1j * torch.angle(ip)).detach()
        mu = (es * w[..., None]).mean(2)
    return mu


def train_protonet_hist(trunk, sampler, n_ep, objective="euclid", unit=True,
                        lr=1e-3, bs=8, device=DEV, log_every=200, tag=""):
    """B17 train_protonet + loss 轨迹返回（SC1 判定用）。"""
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
        if unit:
            es, eq = unit_norm(es), unit_norm(eq)
        mu = es.mean(2)
        if objective == "euclid":
            d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        elif objective == "l1pca":
            # B19-第二修复：轨道对齐（L1-PCA）原型——治"逐帧随机相位 × 相干均值"相消
            mu_o = unit_norm(l1pca_proto(es))
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu_o)
            d2 = 2.0 - 2.0 * ip.abs()
        else:
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


def embed(trunk, Z, device=DEV, bs=1024):
    shape = Z.shape
    flat = Z.reshape(-1, shape[-1])
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            zt = torch.as_tensor(flat[i:i + bs], dtype=torch.complex64, device=device)
            outs.append(trunk(zt).detach().cpu().numpy())
    return np.concatenate(outs, 0).reshape(shape[:-1] + (outs[0].shape[-1],))


def cnorm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)


def heads_on(Zs, Zq, yq, sigma_true, unit=True):
    if unit:
        Zs, Zq = cnorm(Zs), cnorm(Zq)
    Zs = np.ascontiguousarray(Zs, dtype=np.complex128)
    Zq = np.ascontiguousarray(Zq, dtype=np.complex128)
    out = {}
    out["euclid"] = ((E.cls_euclid(Zq, E.proto_euclid(Zs)) == yq).mean(1)).tolist()
    out["orbital"] = ((E.cls_orbital(Zq, E.proto_orbital(Zs)) == yq).mean(1)).tolist()
    mu_p, aux_p = E.phasemap_em(Zs)
    out["phML"] = ((E.cls_marginal(Zq, mu_p, aux_p) == yq).mean(1)).tolist()
    kap = float(np.mean(aux_p["sigma_th"][:, 0]))
    return out, kap


class SynthSampler:
    """σ_θ=0 合成 episode 采样器（train/eval 接口兼容）。"""
    def __init__(self, mu, K, k, m, sigma, seed, sigma_theta=0.0):
        self.mu, self.K, self.k, self.m, self.sigma = mu, K, k, m, sigma
        self.seed, self.sigma_theta = seed, sigma_theta
        self._off = 0

    def sample(self, n, query_snr=None, inject=None, inj_strength=0.0):
        s0 = self.seed + self._off
        self._off += n
        return S.batch_episodes(self.mu, self.K, self.k, self.m, self.sigma_theta,
                                self.sigma, n, seed=s0)

    def train(self):
        pass

    def eval(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full", "synth", "synth2", "fix"])
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    if a.mode == "smoke":
        a.budget, a.epi_eval, a.sigmas = 30, 20, "0"
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    recs = {"meta": {}, "train_hist": {}, "records": []}
    t00 = time.time()

    def make_trunks():
        return {
            "v0_mean": lambda: ComplexAMC2(init="rand", pooling="mean"),
            "v1_hann": lambda: ComplexAMC2(init="hann", pooling="mean"),
            "v13_gated": lambda: ComplexAMC2(init="hann", pooling="gated"),
            "v2_power": lambda: ComplexAMC2(init="hann", pooling="power"),
            "real": lambda: RealAMC(),
        }

    # ---------------- synth 模式：D4 纯优化对照（σ_θ=0 或 synth2 的 σ_θ=π）----------------
    if a.mode in ("synth", "synth2"):
        st_train = 0.0 if a.mode == "synth" else np.pi
        rng = np.random.RandomState(19)
        mu = S.make_prototypes(5, 128, 0.3, rng)
        sigma = float(np.sqrt(1.0 / 128))
        if a.mode == "synth":
            arms = [("v0_mean", "euclid"), ("v0_mean", "orbital"),
                    ("v1_hann", "euclid"), ("v1_hann", "orbital")]
        else:
            arms = [("v0_mean", "orbital"), ("v0_mean", "l1pca"),
                    ("v1_hann", "l1pca"), ("v13_gated", "l1pca")]
        for arm, obj in arms:
            torch.manual_seed(11); np.random.seed(11)
            trunk = make_trunks()[arm]().to(DEV)
            smp = SynthSampler(mu, 5, 5, 15, sigma, seed=5000, sigma_theta=st_train)
            name = f"syn{'' if a.mode == 'synth' else '2'}_{arm}_{obj}"
            trunk, hist = train_protonet_hist(trunk, smp, a.budget, obj, tag=name)
            recs["train_hist"][name] = hist[::10]
            recs["meta"][name] = {"params": int(trunk.n_params()),
                                  "loss_first50": float(np.mean(hist[:50])),
                                  "loss_last50": float(np.mean(hist[-50:]))}
            cell = {"arm": name, "sigma_train": st_train}
            for st in sigmas:
                smp_te = SynthSampler(mu, 5, 5, 15, sigma,
                                      seed=7100 + int(97 * st), sigma_theta=st)
                Zs, Zq, yq = smp_te.sample(a.epi_eval)
                es = embed(trunk, Zs).astype(np.complex128)
                eq = embed(trunk, Zq).astype(np.complex128)
                h, kap = heads_on(es, eq, yq, st, unit=True)
                cell[f"acc_sigma{st:.2f}"] = {k2: float(np.mean(v)) for k2, v in h.items()}
                cell[f"kappa{st:.2f}"] = kap
            if arm != "real":
                cell["cancel_ratio"] = cancellation_ratio(
                    trunk, mu[0][None, :] + sigma * (np.random.randn(8, 128) + 1j * np.random.randn(8, 128)))
            recs["records"].append(cell)
            print(f"[{name}] " + " ".join(f"σ={st:.2f}:{cell[f'acc_sigma{st:.2f}']}"
                  for st in sigmas), flush=True)
        json.dump(recs, open(os.path.join(a.out, "logs", f"B19_{a.mode}.json"), "w"), indent=1)
        print(f"B19[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)
        return

    # ---------------- smoke/full：自然 IQ ----------------
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]
    print(f"=== split0 train={sp['train']} test={sp['test']} ===", flush=True)
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000, snr_min=6, snr_max=18)
    arms = [("v0_mean", "euclid"), ("v0_mean", "orbital"),
            ("v1_hann", "euclid"), ("v1_hann", "orbital"),
            ("v13_gated", "euclid"), ("v13_gated", "orbital"),
            ("v2_power", "euclid"), ("real", "euclid")]
    if a.mode == "fix":
        # B19 第二修复：L1-PCA 原型目标（v0_mean_orbital 朴素均值原型为钉死对照）
        arms = [("v0_mean", "l1pca"), ("v1_hann", "l1pca"), ("v13_gated", "l1pca"),
                ("v2_power", "euclid"), ("v0_mean", "orbital")]
    if a.mode == "smoke":
        arms = arms[:4]
    trunks = {}
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
    probe_y = y[np.random.RandomState(5).choice(len(y), 256, replace=False)]
    for arm, obj in arms:
        torch.manual_seed(11); np.random.seed(11)
        trunk = make_trunks()[arm]().to(DEV)
        name = f"{arm}_{obj}"
        trunk, hist = train_protonet_hist(trunk, smp_tr, a.budget, obj,
                                          log_every=max(a.budget // 6, 50), tag=name)
        trunks[name] = trunk
        recs["train_hist"][name] = hist[::10]
        recs["meta"][name] = {"params": int(trunk.n_params()),
                              "loss_first50": float(np.mean(hist[:50])),
                              "loss_last50": float(np.mean(hist[-50:]))}
        print(f"  [{name}] loss {recs['meta'][name]['loss_first50']:.3f} -> "
              f"{recs['meta'][name]['loss_last50']:.3f}", flush=True)
    # 诊断量：等变 δ / 相消比 / Fisher（嵌入，自然测试类帧）
    for name, tr in trunks.items():
        if name.startswith(("v0", "v1", "v13")):
            recs["meta"][name]["equiv_delta"] = equivariance_error(tr, probe, device=DEV)
            recs["meta"][name]["cancel_ratio"] = cancellation_ratio(tr, probe)
    # 嵌入 Fisher（用训练类 100 帧/类，衡量嵌入类可辨识性）
    fidx = []
    for c in sp["train"]:
        cand = np.where((y == c) & (snr >= 6))[0]
        fidx.append(np.random.RandomState(7).choice(cand, 100, replace=False))
    fidx = np.concatenate(fidx)
    for name, tr in trunks.items():
        e = embed(tr, zn[fidx])
        recs["meta"][name]["fisher"] = fisher_ratio(e, y[fidx])
    # 评估
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
            cell[f"kappa_{name}"] = kap
        recs["records"].append(cell)
        summ = {k2: 100 * float(np.mean(v)) for k2, v in cell["acc"].items()}
        print(f"  [σ={st:.2f}] " + " ".join(f"{k2}={v:.1f}" for k2, v in sorted(summ.items())),
              flush=True)
    json.dump(recs, open(os.path.join(a.out, "logs", f"B19_{a.mode}.json"), "w"), indent=1)
    print(f"B19[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
