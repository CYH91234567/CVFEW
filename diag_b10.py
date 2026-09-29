"""B10诊断：ProtoNet-CNN低于随机水平是真实类偏移现象还是实现bug？

(a) 同模型在train类episode上评估（管道sanity：应远高于随机）
(b) 测试类episode + BN train-mode（batch统计）嵌入
(c) 测试类episode + BN eval-mode（B10协议，running stats）
若(a)高、(c)低、(b)恢复 => BN running-stats在类偏移下失准（发现本身）；
若(a)也低 => 实现bug。
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe.episodes import EpisodeSampler, make_class_splits
from run_b10 import AmcCNN, proto_train, embed

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    z, y, snr = D.load_radioml("/tmp/cvfe_work/code/radioml_cache.npz")
    zn = D.energy_normalize(z).astype(np.complex64)
    splits = make_class_splits(n_splits=5)
    si = 4                                   # split4（B10中ProtoNet=20%的split）
    sp = splits[si]
    smp_tr_ep = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                               q_per_class=5, seed=4000 + si)
    cnn = proto_train(AmcCNN().to(DEV), smp_tr_ep, 4000, "plain")
    res = {}

    def eval_eps(smp, n=200, bn_mode="eval"):
        Zs, Zq, yq = smp.sample(n)
        Zs_t = torch.as_tensor(Zs, dtype=torch.complex64, device=DEV)
        Zq_t = torch.as_tensor(Zq, dtype=torch.complex64, device=DEV)
        Zs_t = Zs_t / Zs_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        Zq_t = Zq_t / Zq_t.abs().mean(dim=-1).clamp_min(1e-8).unsqueeze(-1) if False else \
            Zq_t / Zq_t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)

        def fw(x):
            if bn_mode == "train":
                cnn.train()                   # BN用batch统计
                with torch.no_grad():
                    o = cnn(x)
                cnn.eval()
                return o
            with torch.no_grad():
                return cnn(x)

        B, N, k, L = Zs_t.shape
        es = fw(Zs_t.reshape(-1, L)).reshape(B, N, k, -1).mean(2)
        eq = fw(Zq_t.reshape(-1, L)).reshape(B, Zq_t.shape[1], -1)
        esn = es.cpu().numpy().astype(np.float64)
        eqn = eq.cpu().numpy().astype(np.float64)
        d2 = ((eqn[:, :, None, :] - esn[:, None, :, :]) ** 2).sum(-1)
        return float((d2.argmin(-1) == yq).mean())

    # (a) train类 episode（sanity）
    smp_a = EpisodeSampler(zn, y, snr, classes=sp["train"], n_way=5, k_shot=5,
                           q_per_class=15, seed=990)
    res["train_classes_eval"] = eval_eps(smp_a)
    # (c) 测试类，BN eval-mode（B10协议）
    smp_c = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                           q_per_class=15, seed=991, snr_min=12, snr_max=12)
    res["test_classes_bn_eval"] = eval_eps(smp_c)
    # (b) 测试类，BN train-mode
    res["test_classes_bn_train"] = eval_eps(smp_c, bn_mode="train")
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open("/tmp/cvfe_work/res/logs/B10_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
