"""B18-合成：规范化（选截面）vs 免截面（轨道/PhaseMAP）——割迹代价的定量刻画。

理论背景（P7，本轮新增）：
  C^p/U(1) = CP^{p-1} 上的主 U(1)-丛（p≥2）第一陈类非零 ⇒ **不存在全局连续截面**。
  故任何规范化规则必然在某条割迹（cut locus）上不连续 ⇒ 低 SNR / 幅度相消处
  规范化相位被噪声主导 ⇒ 精度损失。轨道（L1-PCA/MLE）与 PhaseMAP（边际化 θ）
  不选截面，代价是需要估计/积分一个潜变量。本实验定量刻画二者的胜负边界。

网格：K=5, k=5, m=15, ρ=0.3；p∈{16,64}；σ_θ∈{0,π/6,π/3,π/2,2π/3,π}；噪声 σ∈{0.1,0.3,0.6,1.0}
方法：euclid / canon_{first,mean,max,wmean} / canonX_max（支持侧截面+查询侧免截面）
      / orbital / phasemap_ml（免截面、自适应）/ phasemap_ok（oracle κ 上界）
输出：04_results/logs/B18_synth.json（逐 episode 精度）
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth, estim as E

METHODS = ["euclid", "canon_first", "canon_mean", "canon_max", "canon_wmean",
           "canonX_max", "canon_ref", "canonX_ref", "orbital", "phasemap_ml", "phasemap_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ps", default="16,64")
    ap.add_argument("--noises", default=None)
    ap.add_argument("--ratios", default="0.25,1,2,4",
                    help="噪声/信号能量比 r：σ = sqrt(r/p)（使不同 p 可比）")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out = a.out or os.path.join(base, "04_results", "logs", "B18_synth.json")
    sigmas_th = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
    ratios = [float(x) for x in a.ratios.split(",")]
    ps = [int(x) for x in a.ps.split(",")]
    recs = []
    t0 = time.time()
    for p in ps:
        for r in ratios:
            noise = float(np.sqrt(r / p)) if a.noises is None else float(a.noises)
            for st in sigmas_th:
                mu = synth.make_prototypes(a.K, p, a.rho, np.random.RandomState(7 + p))
                acc = {m_: [] for m_ in METHODS}
                for sd in range(a.seeds):
                    Zs, Zq, yq = synth.batch_episodes(mu, a.K, a.k, a.m, st, noise,
                                                      a.E, seed=1000 * (sd + 1) + p)
                    for m_ in METHODS:
                        pred, _ = E.run_method(m_, Zs, Zq, mu_true=mu,
                                               sigma_th_oracle=max(st, 1e-3))
                        acc[m_].extend(((pred == yq).mean(1)).tolist())
                rec = {"p": p, "noise": noise, "ratio": r, "sigma_th": float(st), "K": a.K,
                       "k": a.k, "m": a.m, "rho": a.rho, "n_ep": a.E * a.seeds,
                       "acc": {m_: float(np.mean(v)) for m_, v in acc.items()}}
                recs.append(rec)
                print(f"p={p} r={r:g} σ_θ={st:.2f} | " +
                      " ".join(f"{m_}={100*rec['acc'][m_]:.1f}" for m_ in METHODS), flush=True)
    json.dump({"args": vars(a), "records": recs}, open(out, "w"), indent=1)
    print(f"B18-synth DONE in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
