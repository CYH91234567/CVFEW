"""B34：第二真实数据集（RML2016.10b）验证论文 A 的真实数据三主张。

目的（A-07 的"第二数据集"半边 + 审稿 A-04 的正面化）：
  (1) Thm1 实数据验证可迁移：PhaseMAP/轨道在注入下精确平坦 vs 欧氏下降；
  (2) 监督 SOTA 相位免疫可迁移且跨架构家族（CNN 与 CLDNN 均注入不敏感）；
  (3) 零训练 vs 监督的差距量级可迁移。
协议：
  - 确定性方法（无学习，无泄漏风险）→ 全 10 类 5-way 5-shot（对齐 T2 口径）；
  - SOTA（需训练）→ 防泄漏划分 5/2/3（固定分层），test 类 3-way 5-shot
    （对齐 T5 口径），b=4000，plain。
  - SNR {0,6,12,18} × 注入 {none, π/6, π/2}。
数据：镜像重建（dannis999/RML2016.10b，DeepSig 文本格式），非原始副本。
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler
from run_b10 import AmcCNN, AmcCLDNN, proto_train, embed

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# CLASSES_B 顺序见 make_cache_b10b.py：0=8PSK 1=AM-DSB 2=BPSK 3=CPFSK 4=GFSK
# 5=PAM4 6=QAM16 7=QAM64 8=QPSK 9=WBFM
# 固定分层划分（可复现）：test 含强相位类 8PSK/QAM64 + 模拟 AM-DSB，
# train 5 类用于 SOTA 元训练（5-way 正好），val 2 类作缓冲。
SPLIT_B = {
    "train": [2, 3, 4, 5, 6],      # BPSK, CPFSK, GFSK, PAM4, QAM16
    "val": [8, 9],                 # QPSK, WBFM
    "test": [0, 7, 1],             # 8PSK, QAM64, AM-DSB
}
ALL_CLASSES_B = list(range(10))
SNR_LEVELS = [0, 6, 12, 18]
INJ_GRID = [("none", 0.0), ("global", np.pi / 6), ("global", np.pi / 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/cvfe_work/code/radioml_cache_b10b.npz")
    ap.add_argument("--out", default="/tmp/cvfe_work/res")
    ap.add_argument("--epi-eval", type=int, default=600)
    ap.add_argument("--sota-budget", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    print(f"10b cache: z{z.shape} classes={len(set(y.tolist()))}", flush=True)

    results = []
    t0 = time.time()
    # === (A) 确定性方法：全 10 类 5-way 5-shot（对齐 T2 口径）===
    for sl in SNR_LEVELS:
        for mode, strength in INJ_GRID:
            smp = EpisodeSampler(zn, y, snr, classes=ALL_CLASSES_B, n_way=5, k_shot=5,
                                 q_per_class=15, seed=7000 + 7 * sl,
                                 snr_min=sl, snr_max=sl)
            inj = None if mode == "none" else (mode, strength)
            n = a.epi_eval
            Zs, Zq, yq = smp.sample(n, inject=inj)
            Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
            acc = {}
            acc["euclid"] = float((E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean(1).mean())
            acc["orbital"] = float((E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean(1).mean())
            acc_pm = np.zeros(n)
            for g0 in range(0, n, 100):
                sl_ = slice(g0, min(g0 + 100, n))
                mu, aux = E.phasemap_em(Zs128[sl_])
                acc_pm[sl_] = (E.cls_marginal(Zq128[sl_], mu, aux) == yq[sl_]).mean(1)
            acc["phasemap_ml"] = float(acc_pm.mean())
            rec = {"dataset": "RML2016.10b", "split": "all10", "snr": sl,
                   "inject_mode": mode, "inject_strength": float(strength),
                   "acc": acc, "n_eval": n, "n_way": 5, "k_shot": 5}
            results.append(rec)
            print(f"  [10b all10 SNR{sl} {mode}{strength:.2f}] " +
                  " ".join(f"{k}={100*v:.1f}" for k, v in acc.items()), flush=True)

    # === (B) SOTA：防泄漏划分，test 3-way 5-shot（对齐 T5 口径）===
    smp_tr = EpisodeSampler(zn, y, snr, classes=SPLIT_B["train"], n_way=5, k_shot=5,
                            q_per_class=5, seed=8000)
    for arch, cls in [("cnn", AmcCNN), ("cldnn", AmcCLDNN)]:
        tag = f"sota_protonet_{arch}"
        trunk = proto_train(cls().to(DEV), smp_tr, a.sota_budget, "plain")
        for sl in SNR_LEVELS:
            for mode, strength in INJ_GRID:
                smp_te = EpisodeSampler(zn, y, snr, classes=SPLIT_B["test"], n_way=3,
                                        k_shot=5, q_per_class=15,
                                        seed=9000 + 7 * sl, snr_min=sl, snr_max=sl)
                inj = None if mode == "none" else (mode, strength)
                Zs, Zq, yq = smp_te.sample(400, inject=inj)
                es = embed(trunk, Zs).astype(np.float64)
                eq = embed(trunk, Zq).astype(np.float64)
                d2 = ((eq[:, :, None, :] - es.mean(2)[:, None, :, :]) ** 2).sum(-1)
                accv = float((d2.argmin(-1) == yq).mean())
                results.append({"dataset": "RML2016.10b", "split": "fixed_5/2/3",
                                "arch": arch, "snr": sl, "inject_mode": mode,
                                "inject_strength": float(strength), "acc": {tag: accv},
                                "n_eval": 400, "n_way": 3, "k_shot": 5})
                print(f"  [10b SOTA-{arch} SNR{sl} {mode}{strength:.2f}] {100*accv:.1f}",
                      flush=True)

    path = os.path.join(a.out, "logs", "B34_b10b.json")
    json.dump(results, open(path, "w"), indent=1)
    print(f"B34 DONE in {(time.time()-t0)/60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
