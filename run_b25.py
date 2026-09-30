"""B25：SC-A 第三次攻坚——模型选择 × 容量-预算交互 × 软 γ × 对齐截断（服务器 3090）。

PREREG_B25。六臂 × 种子 {11,23,37}（种子同时变 trunk 初始化与训练流）：
  base1200    l1pca5 @1200            （=B22b cx1200_l1pca5 复现锚点）
  es3200      l1pca5 @3200+验证早停
  big1200     (48,96,192,48) @1200
  big_es3200  big + 早停 @3200
  soft1200    软γ(π/6) @1200，前200 episode 硬 l1pca5 热启动（治 B22 γ→0 不动点）
  l1pca2_1200 l1pca 2 迭代 @1200      （截断强度轴，B26 对应物）
软 γ = run_b22.gamma_soft（wrapN(0,π/6) 先验网格后验，stop-gradient）+ 类内尺度归一。
评估与 B19/B22 严格同协议（split0 测试类 3-way，σ_θ∈{0,π/3,π}，200 epi，三头）。
输出：logs/B25_sca.json（每臂×种子增量落盘，支持断点续跑）。
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm, equivariance_error
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, fisher_ratio
from run_b19 import l1pca_proto, embed, heads_on
from run_b22 import gamma_soft

DEV = "cuda" if torch.cuda.is_available() else "cpu"

ARMS = {
    "base1200":    dict(ch=(32, 64, 128, 32), budget=1200, mode="l1pca5", es=False),
    "es3200":      dict(ch=(32, 64, 128, 32), budget=3200, mode="l1pca5", es=True),
    "big1200":     dict(ch=(48, 96, 192, 48), budget=1200, mode="l1pca5", es=False),
    "big_es3200":  dict(ch=(48, 96, 192, 48), budget=3200, mode="l1pca5", es=True),
    "soft1200":    dict(ch=(32, 64, 128, 32), budget=1200, mode="soft",   es=False),
    "l1pca2_1200": dict(ch=(32, 64, 128, 32), budget=1200, mode="l1pca2", es=False),
}
SEEDS = (11, 23, 37)
SEED_STREAM_OFF = {11: 0, 23: 12, 37: 26}      # 训练流 seed = 4000 + off


def proto_soft(es, st, n_es=3):
    """软 γ 原型：3 内步 E 步（γ stop-gradient，类内尺度归一防 γ→0 数值退化）。"""
    mu = es.mean(2)
    for _ in range(n_es):
        ip = torch.einsum("bnc,bnkc->bnk", mu.conj(), es)
        g = gamma_soft(ip, st).detach()
        g = g / g.abs().mean(-1, keepdim=True).clamp_min(1e-6)
        mu = (es * g[..., None]).sum(2)
    return mu


def make_proto(mode, soft_after=0):
    """返回 fn(es, it)：训练第 it 个 episode 的原型目标。前 soft_after 个 episode
    用硬 l1pca5（对称性破缺引擎），之后软 γ 接管（MMSE 收缩）。"""
    if mode == "l1pca5":
        return lambda es, it: l1pca_proto(es, 5)
    if mode == "l1pca2":
        return lambda es, it: l1pca_proto(es, 2)
    if mode == "soft":
        return lambda es, it: (l1pca_proto(es, 5) if it < soft_after
                               else proto_soft(es, np.pi / 6))
    raise KeyError(mode)


def val_orbital_acc(trunk, val_sampler, n_ep, device=DEV):
    """验证 episode 上的 orbital 头精度（模型选择用，不参与训练）。
    嵌入时切 eval 模式：ModBN 用 running 统计，与最终评估协议一致，且不污染统计。"""
    was_training = trunk.training
    trunk.eval()
    Zs, Zq, yq = val_sampler.sample(n_ep)
    es = embed(trunk, Zs, device=device).astype(np.complex128)
    eq = embed(trunk, Zq, device=device).astype(np.complex128)
    if was_training:
        trunk.train()
    es = es / np.maximum(np.linalg.norm(es, axis=-1, keepdims=True), 1e-12)
    eq = eq / np.maximum(np.linalg.norm(eq, axis=-1, keepdims=True), 1e-12)
    pred = E.cls_orbital(eq, E.proto_orbital(es))
    return float((pred == yq).mean())


def train_arm(trunk, smp_tr, n_ep, mode, es=False, lr=1e-3, bs=8, device=DEV,
              es_every=100, es_patience=5, es_min=600, val_n=100,
              val_sampler=None, log_every=200, tag=""):
    """统一训练循环；es=True 时每 es_every episode 验证一次并 restore 最优。"""
    import torch.nn.functional as F
    proto = make_proto(mode, soft_after=200)
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    hist, val_curve = [], []
    best_val, best_state, best_it, bad, stopped = -1.0, None, 0, 0, n_ep
    t0 = time.time()
    for it in range(n_ep):
        Zs, Zq, yq = smp_tr.sample(bs)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        es_emb = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1))
        mu = unit_norm(proto(es_emb, it))
        ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
        d2 = 2.0 - 2.0 * ip.abs()
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.item()))
        if es and (it + 1) % es_every == 0:
            v = val_orbital_acc(trunk, val_sampler, val_n, device)
            val_curve.append([it + 1, v])
            if v > best_val:
                best_val, best_it, bad = v, it + 1, 0
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
            else:
                bad += 1
            if (it + 1) >= es_min and bad >= es_patience:
                stopped = it + 1
                print(f"    [{tag}] early stop @{it+1} best_val={best_val:.3f}@{best_it}",
                      flush=True)
                break
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    if es and best_state is not None:
        trunk.load_state_dict(best_state)
    trunk.eval()
    info = {"stopped_at": stopped if es else n_ep,
            "best_val": best_val, "best_it": best_it, "val_curve": val_curve,
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:]))}
    return trunk, hist, info


def eval_cell(trunk, zn, y, snr, sp_test, sigmas, epi_eval):
    out = {}
    for st in sigmas:
        smp_te = EpisodeSampler(zn, y, snr, classes=sp_test, n_way=3, k_shot=5,
                                q_per_class=15, seed=7100 + int(97 * st),
                                snr_min=6, snr_max=18)
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp_te.sample(epi_eval, inject=inj)
        es = embed(trunk, Zs).astype(np.complex128)
        eq = embed(trunk, Zq).astype(np.complex128)
        h, kap = heads_on(es, eq, yq, st, unit=True)
        out[f"sigma_{st:.2f}"] = {kk: float(np.mean(v)) for kk, v in h.items()}
        out[f"kappa_{st:.2f}"] = kap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=200)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]

    arms, seeds = ARMS, SEEDS
    if a.mode == "smoke":
        arms = {k: dict(v) for k, v in ARMS.items() if k in
                ("base1200", "soft1200", "es3200")}
        for v in arms.values():
            v["budget"] = 40
        seeds = (11,)
        a.epi_eval, sigmas = 20, [0.0]
        es_cfg = dict(es_every=10, es_patience=2, es_min=20, val_n=10)
    else:
        es_cfg = dict(es_every=100, es_patience=5, es_min=600, val_n=100)

    res_path = os.path.join(a.out, "logs", "B25_sca.json")
    recs = {"meta": {"arms": {k: {kk: vv for kk, vv in v.items()} for k, v in arms.items()},
                     "seeds": list(seeds), "epi_eval": a.epi_eval},
            "runs": {}}
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        recs["runs"].update(old.get("runs", {}))
        print(f"[resume] {len(recs['runs'])} runs already done", flush=True)

    t00 = time.time()
    for arm, cfg in arms.items():
        for seed in seeds:
            key = f"{arm}_s{seed}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            trunk = ComplexAMC2(init="hann", pooling="gated", ch=cfg["ch"]).to(DEV)
            smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                    k_shot=5, q_per_class=5,
                                    seed=4000 + SEED_STREAM_OFF[seed],
                                    snr_min=6, snr_max=18)
            val_smp = None
            if cfg["es"]:
                val_smp = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                         k_shot=5, q_per_class=5,
                                         seed=99000 + seed, snr_min=6, snr_max=18)
            print(f"[{key}] budget={cfg['budget']} mode={cfg['mode']} es={cfg['es']}",
                  flush=True)
            trunk, hist, info = train_arm(
                trunk, smp_tr, cfg["budget"], cfg["mode"], es=cfg["es"],
                val_sampler=val_smp, tag=key, **es_cfg)
            probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
            info["equiv_delta"] = equivariance_error(trunk, probe, device=DEV)
            info["cancel_ratio"] = cancellation_ratio(trunk, probe)
            fidx = []
            for c in sp["train"]:
                cand = np.where((y == c) & (snr >= 6))[0]
                fidx.append(np.random.RandomState(7).choice(cand, 100, replace=False))
            info["fisher"] = fisher_ratio(
                embed(trunk, zn[np.concatenate(fidx)]),
                y[np.concatenate(fidx)])
            cell = eval_cell(trunk, zn, y, snr, sp["test"], sigmas, a.epi_eval)
            recs["runs"][key] = {"info": info, "hist": hist[::20], "eval": cell,
                                 "params": int(trunk.n_params())}
            for st in sigmas:
                acc = cell.get(f"sigma_{st:.2f}", {})
                print(f"  [{key}] σ={st:.2f}: " +
                      " ".join(f"{kk}={100*vv:.1f}" for kk, vv in sorted(acc.items())),
                      flush=True)
            json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B25[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
