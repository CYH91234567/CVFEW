"""B35：负→正攻坚的服务器版实验（split-B 绝对阈值 + real 对照臂 + 中分辨率长滞后读出）。

预注册：`02_plan/PREREG_B35.md`（2026-10-03 写定，服务器恢复后执行）。
本地探针（已完成的先决证据）：`03_code/run_b35_probe.py` →
`04_results/logs/B35_probe.json`、`B35_probe_verdict.json`。

臂：
- `real`    RealAMC 非等变实值对照（euclid 头），B32 协议逐字一致 —— B33 缺失的
            关键对照臂（split-B 上复合 vs real 的配对差是"matching+可靠性"叙事
            在 split-B 成立的判定依据）。
- `midres`  ComplexAMC2 变体：2 级池化（128→32 步，非 3 级→16 步）+ invpow_metric
            读出（tau_max=15）+ 学习度量头。依据探针 F1+F2：长滞后增益在
            中分辨率层级而非 16 步尺度（16 步尺度宽滞后 −5pt）。

【纪律】本脚本含 torch 训练，**必须在服务器端执行**（本地 Windows 无 torch，
且项目纪律：训练一律服务器端）。smoke 模式 60 步预算可用于启动确认（独立
--out 目录，防 B32 式同名污染）。

用法（服务器）：
  python run_b35.py --mode smoke --out /tmp/cvfe_work/res_b35smoke
  python run_b35.py --mode full  --out /tmp/cvfe_work/res_b35
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D  # noqa: E402
from cvfe.episodes import EpisodeSampler, make_class_splits, MOD_CLASSES  # noqa: E402
from cvfe.nets_b17 import unit_norm, RealAMC  # noqa: E402
from cvfe.nets_b19 import ComplexAMC2  # noqa: E402

# 与 run_b32.py 复用同一套协议件（import 其辅助函数，不修改其行为）
from run_b32 import (INIT_SEEDS, SIG, DEV, eval_cell_paired,  # noqa: E402
                     make_train_val_samplers, quick_val_train,
                     new_ema, ema_update, ema_loadable)
from cvfe.nets_b17 import equivariance_error  # noqa: E402


class MidResComplexAMC(ComplexAMC2):
    """B35 中分辨率变体：hidden() 只做 2 级 cpool2（128→64→32 步），
    保留更长的时间轴供长滞后不变统计（tau_max 由构造传入，默认 15）。
    等变性与不变性性质与父类同（复卷积 + ModBN + modReLU 均相位等变），
    逐点不变性由 invariant_stats 保证。"""

    def hidden(self, z):
        import torch.nn as nn  # noqa: F401  (与父类实现风格一致)
        from cvfe.nets_b19 import cpool2
        h = z[:, None, :]
        h = cpool2(self.a1(self.b1(self.c1(h))))
        h = cpool2(self.a2(self.b2(self.c2(h))))
        h = self.a3(self.b3(self.c3(h)))          # 第 3 级不池化 → 32 步
        return self.b4(self.c4(h))


def cancellation_ratio(trunk, probe, device, arm):
    """与 run_b32 同口径的末端相干比诊断（机制可观测代理）。
    仅对有时时间轴隐层的臂（midres）有意义；RealAMC 的 forward 已在时间维取均值
    （无时间轴）⇒ 返回 None。"""
    if arm == "real":
        return None
    trunk.eval()
    with torch.no_grad():
        z = torch.as_tensor(probe, dtype=torch.complex64, device=device)
        h = trunk.hidden(z)
        a2 = h.abs() ** 2
        return float((a2.mean(-1).min(-1).values / a2.mean(-1).max(-1).values).mean().item())


def train_one_b35(arm, i0, zn, y, snr, sp, budget, val_every, val_n, epi_eval):
    """与 run_b32.train_one 同协议；arm ∈ {"real", "midres"}。"""
    import torch.nn.functional as F
    torch.manual_seed(i0)
    np.random.seed(i0)
    if arm == "real":
        trunk = RealAMC().to(DEV)
    else:
        trunk = MidResComplexAMC(init="hann", pooling="invpow_metric",
                                 ch=(48, 96, 192, 48), tau_max=15).to(DEV)
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
        # real 与 midres 输出均为实特征（RealAMC 实值；midres 的 metric 头输出实向量）
        # ⇒ 均值原型 + 欧氏（与 INVARIANT_ARMS 同口径）
        mu = unit_norm(es.mean(2))
        d2 = ((eq.unsqueeze(2) - mu.unsqueeze(1)).abs() ** 2).sum(-1)
        loss = F.cross_entropy(-d2.reshape(-1, N),
                                torch.as_tensor(yq.reshape(-1), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(ema_state, trunk)
        hist.append(float(loss.item()))
        if (it + 1) % val_every == 0:
            vt = quick_val_train(trunk, val_smps, val_n, arm="invpow_metric")
            if vt > best_val:
                best_val, best_it = vt, it + 1
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in trunk.state_dict().items()}
    trunk.eval()
    probe = zn[np.random.RandomState(5).choice(len(zn), 256, replace=False)]
    info = {"best_val": best_val, "best_it": best_it,
            "train_min": (time.time() - t0) / 60,
            "cancel_ratio": cancellation_ratio(trunk, probe, DEV, arm),
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:])),
            "params": int(sum(p.numel() for p in trunk.parameters()))}
    # real 非等变 ⇒ 只有 midres 报 invariance_delta（构造性不变证书）
    if arm == "midres":
        from cvfe.nets_b19 import invariance_error
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
        head = "euclid"
        accs = {s: 100 * ev[name][f"sigma_{s:.2f}"][head] for s in SIG}
        print(f"  [{arm}|i{i0}|{name}] " +
              " ".join(f"σ={s:.2f}:{v:.1f}" for s, v in accs.items()) +
              f"  mean={np.mean(list(accs.values())):.2f}", flush=True)
        del twin
    return info, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res_b35")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--epi-eval", type=int, default=500)
    ap.add_argument("--arms", default="real,midres")
    ap.add_argument("--budget", type=int, default=3200)
    ap.add_argument("--split-seed", type=int, default=777,
                    help="split-B（与 B33 一致）")
    ap.add_argument("--split-idx", type=int, default=1)
    ap.add_argument("--n-inits", type=int, default=8,
                    help="B33 协议 n=8；init 集 INIT_SEEDS 前 8")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    sp = make_class_splits(n_splits=a.split_idx + 1, seed=a.split_seed)[a.split_idx]

    arms = a.arms.split(",")
    if a.mode == "smoke":
        budget, val_every, val_n, epi_eval = 60, 20, 10, 20
    else:
        budget, val_every, val_n, epi_eval = a.budget, 100, 100, a.epi_eval

    for arm in arms:
        res_path = os.path.join(a.out, "logs", f"B35_{arm}.json")
        recs = {"meta": {"prereg": "PREREG_B35.md (split-B 攻坚 + real 对照 + 中分辨率)",
                         "arm": arm, "budget": budget,
                         "inits": list(INIT_SEEDS[:a.n_inits]),
                         "n_inits": a.n_inits,
                         "split_seed": a.split_seed, "split_idx": a.split_idx,
                         "split_test_classes": [MOD_CLASSES[c] for c in sp["test"]],
                         "stream_seed": 4000,
                         "eval": "sigma-paired (same base episodes across sigma)",
                         "disjoint": True, "deterministic": True,
                         "protocol": "B32/B33 identical; probe: B35_probe_verdict.json",
                         "comp_arm_ref": "B33_invpow_metric.json (already complete, n=8)"},
                "runs": {}}
        if os.path.exists(res_path):
            old = json.load(open(res_path))
            recs["runs"].update(old.get("runs", {}))
            recs["meta"] = old.get("meta", recs["meta"])
            print(f"[resume] {res_path}: {len(recs['runs'])} done", flush=True)
        for i0 in INIT_SEEDS[:a.n_inits]:
            key = f"{arm}_i{i0}"
            if key in recs["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            print(f"[run] {key}", flush=True)
            info, ev = train_one_b35(arm, i0, zn, y, snr, sp, budget,
                                     val_every, val_n, epi_eval)
            recs["runs"][key] = {"info": info, "eval": ev}
            with open(res_path, "w") as f:
                json.dump(recs, f)
        print(f"[done] {arm}: {res_path}", flush=True)


if __name__ == "__main__":
    main()
