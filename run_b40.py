"""B40：第二真实数据集（RML2016.10b）学习层四臂（gap B-J1）。

论文 B 的学习层证据此前全部在 RML2016.10a（B33 为同集类不相交分割，不算第二
数据集；gap-tsp-tnnls B-J1 major）。本实验把 B32/B33/B35 的四臂面板逐字迁移到
RML2016.10b（固定 5/2/3 分层划分，与 B34 的 10b SOTA 半边同 split：
test = {8PSK, QAM64, AM-DSB} 3-way 5-shot），检验四条主张的跨数据集复现：
  (1) 复合不变读出相对 gated 等变学习臂的配对增益（10a split-0 +16.01pt /
      split-B +33.67pt）在第二真实数据集是否保持为正（判据放宽到 +5pt）；
  (2) 复合/midres 臂与非等变 real 对照的持平（parity）是否复现（real 依赖增广
      吸收相位，复合构造性不变）；
  (3) midres（中分辨率长滞后读出）相对 16 步复合臂的方向性增益是否复现；
  (4) 构造性不变证书（δ_inv float 级、决策级 σ 平坦）跨数据集成立——构造性
      与数据无关，理论必过，实测确认。
协议与 B32/B33/B35（10a）逐字一致：INIT_SEEDS[0:8]（与 B33/B35 同 n=8，统计
功效如实标注）× 流 4000 × 3200 步 × valsel/ema/endpoint 三头 × σ 配对评估
（SIG={0,π/3,π}，500 epi/σ）。判据 PREREG_B40.md（运行前写定）。

臂：
- real    RealAMC 非等变实值对照（euclid 头）
- gated   ComplexAMC2(pooling="gated") 等变学习臂 + l1pca 轨道头（B30 基线）
- comp    ComplexAMC2(pooling="invpow_metric") 复合不变臂（B32/B33 主臂）
- midres  MidResComplexAMC（2 级池化 128→32 步 + invpow_metric tau_max=15，B35）

【纪律】本脚本含 torch 训练，**必须在服务器端执行**（本地 Windows 无 torch）。
smoke 模式 60 步预算可用于启动确认（独立 --out 目录，防 B32 式同名污染）。
每个臂写独立 json（B40_{arm}.json），故四臂可作四个并行单臂进程安全运行。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, copy, json, sys, time

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D  # noqa: E402
from cvfe.episodes import EpisodeSampler  # noqa: E402
from cvfe.nets_b17 import unit_norm, RealAMC, equivariance_error  # noqa: E402
from cvfe.nets_b19 import ComplexAMC2, invariance_error  # noqa: E402
from run_b32 import (INIT_SEEDS, SIG, DEV, eval_cell_paired,  # noqa: E402
                     make_train_val_samplers, quick_val_train,
                     new_ema, ema_update, ema_loadable)
from run_b19 import l1pca_proto  # noqa: E402
from run_b35 import MidResComplexAMC, cancellation_ratio  # noqa: E402

# 10b 固定分层划分（与 B34 的 10b 半边同 split；CLASSES_B 顺序见 make_cache_b10b.py：
# 0=8PSK 1=AM-DSB 2=BPSK 3=CPFSK 4=GFSK 5=PAM4 6=QAM16 7=QAM64 8=QPSK 9=WBFM）
SPLIT_B10 = {"train": [2, 3, 4, 5, 6],   # BPSK, CPFSK, GFSK, PAM4, QAM16
             "val": [8, 9],              # QPSK, WBFM
             "test": [0, 7, 1]}          # 8PSK, QAM64, AM-DSB（3-way）
CLASS_NAMES_B10 = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4",
                   "QAM16", "QAM64", "QPSK", "WBFM"]
ARM_HEAD = {"real": "euclid", "gated": "orbital",
            "comp": "euclid", "midres": "euclid"}
COMPLEX_OUT = {"gated"}   # 复向量输出（等变 ⇒ 轨道头）；其余三臂输出实向量
# quick_val_train 的 arm 别名：复输出臂走 orbital，实输出臂强制 euclid
QUICK_ARM = {"gated": "gated", "real": "invpow_metric",
             "comp": "invpow_metric", "midres": "invpow_metric"}


def make_trunk(arm):
    if arm == "real":
        return RealAMC().to(DEV)
    if arm == "gated":
        return ComplexAMC2(init="hann", pooling="gated",
                           ch=(48, 96, 192, 48)).to(DEV)
    if arm == "comp":
        return ComplexAMC2(init="hann", pooling="invpow_metric",
                           ch=(48, 96, 192, 48)).to(DEV)
    if arm == "midres":
        return MidResComplexAMC(init="hann", pooling="invpow_metric",
                                ch=(48, 96, 192, 48), tau_max=15).to(DEV)
    raise ValueError(arm)


def train_one_b40(arm, i0, zn, y, snr, sp, budget, val_every, val_n, epi_eval):
    """与 run_b32.train_one / run_b35.train_one_b35 同协议；四臂头部分派。"""
    import torch.nn.functional as F
    torch.manual_seed(i0)
    np.random.seed(i0)
    trunk = make_trunk(arm)
    smp_tr = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=4000, snr_min=6, snr_max=18)
    val_smps = make_train_val_samplers(zn, y, snr, sp, 88000)
    opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3, weight_decay=1e-4)
    trunk.train()
    hist = []
    ema_state = new_ema()
    best_val, best_state, best_it = -1.0, None, 0
    t0 = time.time()
    for it in range(budget):
        Zs, Zq, yq = smp_tr.sample(8)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        B, N, k, L = Zs_t.shape
        m = Zq_t.shape[1]
        es = unit_norm(trunk(Zs_t.reshape(-1, L)).reshape(B, N, k, -1))
        eq = unit_norm(trunk(Zq_t.reshape(-1, L)).reshape(B, m, -1))
        if arm in COMPLEX_OUT:
            mu = unit_norm(l1pca_proto(es, 5))
            ip = torch.einsum("bmc,bnc->bmn", eq.conj(), mu)
            d2 = 2.0 - 2.0 * ip.abs()
        else:
            mu = unit_norm(es.mean(2))
            d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        loss = F.cross_entropy(-d2.reshape(-1, N),
                                torch.as_tensor(yq.reshape(-1), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(ema_state, trunk)
        hist.append(float(loss.item()))
        if (it + 1) % val_every == 0:
            vt = quick_val_train(trunk, val_smps, val_n, arm=QUICK_ARM[arm])
            if vt > best_val:
                best_val, best_it = vt, it + 1
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
    trunk.eval()
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
    info = {"best_val": best_val, "best_it": best_it,
            "train_min": (time.time() - t0) / 60,
            "cancel_ratio": (None if arm == "real"
                             else cancellation_ratio(trunk, probe, DEV, arm)),
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:])),
            "params": int(sum(p.numel() for p in trunk.parameters())),
            "arm_head": ARM_HEAD[arm]}
    if arm in ("comp", "midres"):
        info["invariance_delta"] = float(invariance_error(trunk, probe, device=DEV))
    else:
        info["equiv_delta"] = float(equivariance_error(trunk, probe, device=DEV))
    sets = {"endpoint": {kk: vv.detach().cpu().clone()
                         for kk, vv in trunk.state_dict().items()}}
    if best_state is not None:
        sets["valsel"] = best_state
    if ema_state["state"] is not None:
        sets["ema"] = ema_loadable(ema_state, trunk)
    ev = {}
    for name, sd in sets.items():
        twin = copy.deepcopy(trunk)
        twin.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
        twin.eval()
        ev[name] = eval_cell_paired(twin, zn, y, snr, sp["test"], SIG, epi_eval)
        head = ARM_HEAD[arm]
        accs = {s: 100 * ev[name][f"sigma_{s:.2f}"][head] for s in SIG}
        print(f"  [{arm}|i{i0}|{name}] " +
              " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
              f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
        del twin
    return info, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache_b10b.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res_b40")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--arms", default="real,gated,comp,midres")
    ap.add_argument("--budget", type=int, default=3200)
    ap.add_argument("--n-inits", type=int, default=8,
                    help="INIT_SEEDS 前 N 个（与 B33/B35 n=8 同口径）")
    ap.add_argument("--init-offset", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    print(f"10b cache: z{z.shape} classes={len(set(y.tolist()))}", flush=True)
    sp = SPLIT_B10

    arms = a.arms.split(",")
    if a.mode == "smoke":
        budget, val_every, val_n, epi_eval = 60, 20, 10, 20
    else:
        budget, val_every, val_n, epi_eval = a.budget, 100, 100, a.epi_eval

    for arm in arms:
        res_path = os.path.join(a.out, "logs", f"B40_{arm}.json")
        recs = {"meta": {"prereg": "PREREG_B40.md (10b 第二真实数据集学习层四臂, B-J1)",
                         "dataset": "RML2016.10b (mirror rebuild, dannis999/RML2016.10b)",
                         "arm": arm, "budget": budget,
                         "inits": list(INIT_SEEDS[a.init_offset:a.init_offset + a.n_inits]),
                         "n_inits": a.n_inits,
                         "split": "fixed 5/2/3 (same as B34 10b SOTA half)",
                         "split_train_classes": [CLASS_NAMES_B10[c] for c in sp["train"]],
                         "split_val_classes": [CLASS_NAMES_B10[c] for c in sp["val"]],
                         "split_test_classes": [CLASS_NAMES_B10[c] for c in sp["test"]],
                         "stream_seed": 4000,
                         "eval": "sigma-paired (same base episodes across sigma)",
                         "disjoint": True, "deterministic": True,
                         "protocol": "B32/B33/B35 (10a) replicated on 10b",
                         "arm_head": ARM_HEAD[arm]}, "runs": {}}
        if os.path.exists(res_path):
            old = json.load(open(res_path))
            recs["runs"].update(old.get("runs", {}))
            recs["meta"] = old.get("meta", recs["meta"])
            print(f"[resume] {res_path}: {len(recs['runs'])} done", flush=True)
        for i0 in INIT_SEEDS[a.init_offset:a.init_offset + a.n_inits]:
            key = f"{arm}_i{i0}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            print(f"[run] {key}", flush=True)
            info, ev = train_one_b40(arm, i0, zn, y, snr, sp, budget,
                                     val_every, val_n, epi_eval)
            recs["runs"][key] = {"info": info, "eval": ev}
            with open(res_path, "w") as f:
                json.dump(recs, f)
        print(f"[done] {arm}: {res_path}", flush=True)


if __name__ == "__main__":
    main()
