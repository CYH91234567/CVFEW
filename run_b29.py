"""B29：SC-A 第四次攻坚——管道修复版（PREREG_B29，服务器 3090）。

B28 审计确认的五项管道缺陷在本协议中全部修复（详见 PREREG_B29 §0 表）：
  F1 val 训练类错位 → sp["val"] 两个未见类，2-way 5-shot q=15，σ∈{0,π/3,π} 注入均值
  F2 终点快照伪象   → 固定 3200 步 + 每 100 步 checkpoint，事后用对齐 val 选点（不早停）
  F3 EMA 稻草人     → 参数 EMA（排除 ModBN running buffer），decay=0.995，偏置校正
  F4 支持/查询重叠  → EpisodeSampler(disjoint=True)（episodes.py B28 修复，默认开）
  F5 种子捆绑+非确定 → init×流解耦（4×2）+ 确定性开关 + 噪声探针模式

臂：
  cx_v2    等变复 CNN big(48,96,192,48)+hann+gated，l1pca5，3200 步，8 runs（4 init×2 流）
  real_v2  RealAMC 实对照（euclid 目标/头），同协议，3 runs（阈值锚）
  cxpn_v2  cx_v2 + 训练 50% episode 施加 pnoise(0.1)（非全局 nuisance 混合的公平检验），4 runs
评估：测试类 3-way 5-shot q=15，σ∈{0,π/3,π}，500 epi/σ，三头；每 run 评三个权重集
  valsel（对齐 val 选点）/ endpoint / ema。
输出：logs/B29_sca_v2.json（增量落盘，断点续跑）。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, sys, time, copy
import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits
from cvfe.nets_b17 import unit_norm, equivariance_error, RealAMC
from cvfe.nets_b19 import ComplexAMC2, cancellation_ratio, fisher_ratio
from run_b19 import l1pca_proto, embed, heads_on, cnorm

DEV = "cuda" if torch.cuda.is_available() else "cpu"

INIT_SEEDS = (11, 23, 37, 51)
STREAM_OFFS = (0, 12)                      # 训练流 seed = 4000 + off
SIGMAS = (0.0, np.pi / 3, np.pi)
# 测试流 seed = 7100 + int(97σ)（与 B19/B22/B25 评估流同公式）
ARMS = {
    "cx_v2":   dict(kind="cx",  streams=STREAM_OFFS, inits=INIT_SEEDS, pnoise=False),
    "real_v2": dict(kind="real", streams=(0, 12, 26), inits=(11, 23, 37), pnoise=False),
    "cxpn_v2": dict(kind="cx",  streams=(0,), inits=INIT_SEEDS, pnoise=True),
}
PN_STRENGTH = 0.1


# ---------------- EMA（F3 修复：参数 only + 偏置校正 + 排除 running buffer） ----------------
def new_ema(decay=0.995):
    return {"decay": decay, "t": 0, "state": None}


def ema_update(ema, trunk):
    params = {k: v for k, v in trunk.state_dict().items()
              if v.is_floating_point() and "running" not in k}
    if ema["state"] is None:
        ema["state"] = {k: v.detach().clone().float() for k, v in params.items()}
        ema["t"] = 1
        return
    ema["t"] += 1
    d = ema["decay"]
    with torch.no_grad():
        for k, v in params.items():
            ema["state"][k].mul_(d).add_(v.detach().float(), alpha=1 - d)


def ema_loadable(ema, trunk):
    """EMA 偏置校正后的完整 state_dict：参数用 EMA/(1−d^t)，BN 统计复制当前值。"""
    sd = {k: v.detach().clone() for k, v in trunk.state_dict().items()}
    bc = 1.0 - ema["decay"] ** ema["t"]
    for k, v in ema["state"].items():
        sd[k] = (v / bc).to(sd[k].dtype)
    return sd


# ---------------- 对齐 val（F1 修复：未见类 + 注入均值） ----------------
def make_val_samplers(zn, y, snr, sp, n_way, k_shot, q_per_class, val_n_seed0):
    out = []
    for i, st in enumerate(SIGMAS):
        smp = EpisodeSampler(zn, y, snr, classes=sp["val"], n_way=n_way, k_shot=k_shot,
                             q_per_class=q_per_class, seed=val_n_seed0 + 1000 + i,
                             snr_min=6, snr_max=18)
        out.append((smp, st))
    return out


@torch.no_grad()
def val_score(trunk, val_smps, head, n_val, device=DEV):
    """对齐 val 精度。嵌入用 eval 模式（ModBN running 统计，与最终评估一致），
    结束后恢复原模式——训练循环中的中间检查不得让 trunk 停留在 eval。"""
    was_training = trunk.training
    trunk.eval()
    vals = []
    for smp, st in val_smps:
        inj = None if st == 0 else ("global", st)
        Zs, Zq, yq = smp.sample(n_val, inject=inj)
        es = cnorm(embed(trunk, Zs, device=device)).astype(np.complex128)
        eq = cnorm(embed(trunk, Zq, device=device)).astype(np.complex128)
        if head == "orbital":
            pred = E.cls_orbital(eq, E.proto_orbital(es))
        else:
            pred = E.cls_euclid(eq, E.proto_euclid(es))
        vals.append(float((pred == yq).mean()))
    if was_training:
        trunk.train()
    return float(np.mean(vals))


# ---------------- 训练（F2 修复：全程 checkpoint，不早停） ----------------
def train_arm(trunk, smp_tr, n_ep, mode="l1pca5", bs=8, lr=1e-3, device=DEV,
              val_every=100, val_smps=None, val_head="orbital", val_n=200,
              pnoise=False, ema=True, log_every=400, tag=""):
    import torch.nn.functional as F
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, weight_decay=1e-4)
    trunk.train()
    hist, val_curve = [], []
    ema_state = new_ema() if ema else None
    best_val, best_state, best_it = -1.0, None, 0
    t0 = time.time()
    for it in range(n_ep):
        inject = ("pnoise", PN_STRENGTH) if (pnoise and it % 2 == 0) else None
        Zs, Zq, yq = smp_tr.sample(bs, inject=inject)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=device)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=device)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        es = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1))
        if mode == "l1pca5":
            mu = unit_norm(l1pca_proto(es, 5))
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
            d2 = 2.0 - 2.0 * ip.abs()
        elif mode == "euclid":
            mu = es.mean(2)
            d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        else:
            raise KeyError(mode)
        loss = F.cross_entropy(-d2.reshape(-1, N),
                               torch.as_tensor(yq.reshape(-1), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.item()))
        if ema_state is not None:
            ema_update(ema_state, trunk)
        if val_smps is not None and (it + 1) % val_every == 0:
            v = val_score(trunk, val_smps, val_head, val_n, device)
            val_curve.append([it + 1, v])
            if v > best_val:
                best_val, best_it = v, it + 1
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
        if (it + 1) % log_every == 0:
            vb = f" best_val={best_val:.3f}@{best_it}" if val_curve else ""
            print(f"    [{tag}] ep{it+1}/{n_ep} loss={loss.item():.4f} "
                  f"({(time.time()-t0)/60:.1f}min){vb}", flush=True)
    trunk.eval()
    info = {"best_val": best_val, "best_it": best_it, "val_curve": val_curve,
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:]))}
    return trunk, hist, info, best_state, ema_state


# ---------------- 评估 ----------------
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


def run_one(arm, cfg, init_seed, stream_off, zn, y, snr, sp, sigmas, epi_eval,
            budget, val_every, val_n, out_path, recs, smoke=False):
    key = f"{arm}_i{init_seed}_f{stream_off}"
    if key in recs["runs"]:
        print(f"[skip] {key}", flush=True)
        return
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    val_head = "euclid" if cfg["kind"] == "real" else "orbital"
    if cfg["kind"] == "real":
        trunk = RealAMC().to(DEV)
        mode = "euclid"
    else:
        trunk = ComplexAMC2(init="hann", pooling="gated",
                            ch=(48, 96, 192, 48)).to(DEV)
        mode = "l1pca5"
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000 + stream_off,
                            snr_min=6, snr_max=18)
    n_way_v = 2
    val_smps = make_val_samplers(zn, y, snr, sp, n_way_v, 5, 15, 99000)
    print(f"[{key}] kind={cfg['kind']} mode={mode} pnoise={cfg['pnoise']} "
          f"budget={budget}", flush=True)
    trunk, hist, info, best_state, ema_state = train_arm(
        trunk, smp_tr, budget, mode, val_every=val_every, val_smps=val_smps,
        val_head=val_head, val_n=val_n, pnoise=cfg["pnoise"], tag=key,
        log_every=max(budget // 8, 20))
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
    info["equiv_delta"] = (equivariance_error(trunk, probe, device=DEV)
                           if cfg["kind"] == "cx" else None)
    info["cancel_ratio"] = (cancellation_ratio(trunk, probe)
                            if cfg["kind"] == "cx" else None)
    fidx = []
    for c in sp["train"]:
        cand = np.where((y == c) & (snr >= 6))[0]
        fidx.append(np.random.RandomState(7).choice(cand, 100, replace=False))
    info["fisher"] = fisher_ratio(embed(trunk, zn[np.concatenate(fidx)]),
                                  y[np.concatenate(fidx)])
    endpoint_state = {kk: vv.detach().cpu().clone()
                      for kk, vv in trunk.state_dict().items()}
    entry = {"info": info, "hist": hist[::20], "params": int(trunk.n_params()),
             "eval": {}}
    # 三个权重集：valsel（noise/smoke 模式可能无选点）/ endpoint / ema
    sets = {"endpoint": endpoint_state}
    if best_state is not None:
        sets["valsel"] = best_state
    if ema_state is not None and ema_state["state"] is not None:
        sets["ema"] = ema_loadable(ema_state, trunk)
    for name, sd in sets.items():
        twin = copy.deepcopy(trunk)
        twin.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
        twin.eval()
        entry["eval"][name] = eval_cell(twin, zn, y, snr, sp["test"], sigmas, epi_eval)
        accs = {st: 100 * entry["eval"][name][f"sigma_{st:.2f}"][val_head]
                for st in sigmas}
        print(f"  [{key}|{name}] " +
              " ".join(f"σ={st:.2f}:{v:.1f}" for st, v in accs.items()) +
              f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
        del twin
    recs["runs"][key] = entry
    json.dump(recs, open(out_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--mode", default="full", choices=["smoke", "full", "noise"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--sigmas", default="0,1.0471976,3.1415927")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=1)[0]

    if a.mode == "smoke":
        budget, val_every, val_n, a.epi_eval, sigmas = 60, 20, 10, 20, [0.0]
        arms = {k: v for k, v in ARMS.items()}
        for v in arms.values():
            v["inits"], v["streams"] = (11,), (0,)
    elif a.mode == "noise":
        budget, val_every, val_n, a.epi_eval, sigmas = 600, 10 ** 9, 10, 100, [0.0]
        arms = {"cx_v2": dict(ARMS["cx_v2"], inits=(11,), streams=(0,))}
    else:
        budget, val_every, val_n = 3200, 100, 200
        arms = ARMS

    res_path = os.path.join(a.out, "logs", "B29_sca_v2.json")
    recs = {"meta": {"prereg": "PREREG_B29.md", "budget": budget,
                     "val_every": val_every, "val_n": val_n,
                     "epi_eval": a.epi_eval, "init_seeds": list(INIT_SEEDS),
                     "stream_offs": list(STREAM_OFFS), "disjoint": True,
                     "ema_decay": 0.995, "deterministic": True},
            "runs": {}}
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        recs["runs"].update(old.get("runs", {}))
        print(f"[resume] {len(recs['runs'])} runs done", flush=True)
    t00 = time.time()
    for arm, cfg in arms.items():
        for i0 in cfg["inits"]:
            for so in cfg["streams"]:
                run_one(arm, cfg, i0, so, zn, y, snr, sp, sigmas, a.epi_eval,
                        budget, val_every, val_n, res_path, recs,
                        smoke=(a.mode == "smoke"))
    if a.mode == "noise":
        # 噪声探针：同种子完整重跑第二次，报告重跑差
        second = {}
        for key in list(recs["runs"]):
            if key.endswith("_f0") and "_i11" in key:
                k2 = key + "_rerun"
                torch.manual_seed(11); np.random.seed(11)
                trunk = ComplexAMC2(init="hann", pooling="gated",
                                    ch=(48, 96, 192, 48)).to(DEV)
                smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5,
                                        k_shot=5, q_per_class=5, seed=4000,
                                        snr_min=6, snr_max=18)
                trunk, hist, info, _, _ = train_arm(
                    trunk, smp_tr, budget, "l1pca5", tag=k2,
                    log_every=max(budget // 8, 20))
                cell = eval_cell(trunk, zn, y, snr, sp["test"], sigmas, a.epi_eval)
                second[k2] = {"loss_last100": info["loss_last100"],
                              "eval": cell}
                print(f"[noise] {key}: loss {recs['runs'][key]['info']['loss_last100']:.4f}"
                      f" vs {info['loss_last100']:.4f}", flush=True)
        recs["noise_probe"] = second
        json.dump(recs, open(res_path, "w"), indent=1)
    print(f"B29[{a.mode}] DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
