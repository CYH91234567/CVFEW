"""B8：RadioML RML2016.10a 少样本真实IQ实验（估计器级，identity 特征 = 能量归一化原始IQ）。

协议：
  - 5 个 6/2/3 类划分（数字/模拟分层），测试类元训练不可见
  - 注入: none(自然) / global σ_θ∈{π/6, π/2}（导频辅助场景）
  - 5-way 1/5-shot，600 episodes/格，support 与 query 同 SNR 层
  - 跨 SNR 难迁移协议: support@高SNR档 -> query@低SNR档（单独报告）
  - 按 SNR 分层记录（-高/中/低三档）
方法: euclid, orbital, tta_euclid, circcoord, whiten, drop_agc,
      phasemap, phasemap_ml, phasemap_w（9 个）
"""
import argparse, base64, json, os, sys, time, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits

METHODS = ["euclid", "orbital", "tta_euclid", "circcoord", "whiten", "drop_agc",
           "phasemap", "phasemap_ml", "phasemap_w"]


def packbits_b64(arr):
    return base64.b64encode(np.packbits(np.asarray(arr, dtype=np.uint8))).decode()


def eval_episodes(sampler, n_epi, methods, chunk=32, kappa_meta=np.pi / 3,
                  query_snr=None, inject=None):
    acc = {m: [] for m in methods}
    snr_tag = []
    done = 0
    while done < n_epi:
        n = min(chunk, n_epi - done)
        Zs, Zq, yq = sampler.sample(n, query_snr=query_snr, inject=inject)
        Zs = Zs.astype(np.complex128)
        Zq = Zq.astype(np.complex128)
        for mth in methods:
            if mth == "whiten":
                mu = E.proto_euclid(Zs)
                aux = E.whiten_aux(Zq)
                pred = E.cls_mahalanobis_diag(Zq, mu, aux)
            else:
                pred, _ = E.run_method(mth, Zs, Zq, mu_true=None, kappa_meta=kappa_meta)
            acc[mth].extend((pred == yq).mean(axis=1).tolist())
        # episode 的 SNR 档（用 support 的 SNR 层近似——采样器内部决定）
        done += n
    return acc


_G = {}


def _init_worker(g):
    global _G
    _G = g


def _worker_b8(job):
    sl, mode, strength, k = job
    smp = EpisodeSampler(_G["zn"], _G["y"], _G["snr"], classes=_G["all_classes"],
                         n_way=5, k_shot=k, q_per_class=15,
                         snr_min=sl, snr_max=sl, seed=7777 + 31 * sl)
    inj = None if mode == "none" else (mode, strength)
    accs = eval_episodes(smp, _G["epi"], METHODS, kappa_meta=np.pi / 3, inject=inj)
    key = f"snr{sl}|{mode}{strength:.2f}|k{k}"
    return {"cond": key, "snr": int(sl), "inject_mode": mode,
            "inject_strength": float(strength), "k": k,
            "acc": {m: float(np.mean(accs[m])) for m in METHODS},
            "acc_se": {m: float(np.std(accs[m]) / np.sqrt(len(accs[m]))) for m in METHODS},
            "per_episode_b64": {m: packbits_b64(np.array(accs[m]) > 0) for m in METHODS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"D:\个人\CVCNN\CVXAI\04_results_from_server\radioml_cache.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epi", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    from cvfe import data as D
    z, y, snr = D.load_radioml(a.cache)
    zn = D.energy_normalize(z).astype(np.complex64)
    epi = 150 if a.quick else a.epi
    # 估计器级实验无学习成分（kappa_meta 先验固定），episode 直接在全部 11 类上采样；
    # 6/2/3 类划分协议保留给 B7 元学习（meta-learned kappa）使用
    all_classes = list(range(11))
    snr_levels = sorted(set(snr.tolist()))
    inj_grid = [("none", 0.0), ("global", np.pi / 6), ("global", np.pi / 2)]
    jobs = []
    for sl in snr_levels:
        for mode, strength in inj_grid:
            for k in (1, 5):
                jobs.append((sl, mode, strength, k))
    print(f"B8 jobs: {len(jobs)} x {epi} episodes (SNR levels {snr_levels})")

    gdata = {"zn": zn, "y": y, "snr": snr, "all_classes": all_classes, "epi": epi}
    t0 = time.time()
    import multiprocessing as mp
    results = []
    with mp.Pool(processes=min(a.workers, len(jobs)),
                 initializer=_init_worker, initargs=(gdata,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_b8, jobs)):
            results.append(r)
            print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(jobs)} {r['cond']} "
                  + " ".join(f"{m}={100*r['acc'][m]:.1f}" for m in
                             ["euclid", "orbital", "phasemap_ml"]), flush=True)
    results.sort(key=lambda r: r["cond"])
    json.dump(results, open(os.path.join(out, "logs", "B8_radioml.json"), "w"), indent=1)

    # 跨 SNR 难迁移（support 高SNR档 -> query 低SNR档）
    print("cross-SNR protocol (support@18dB -> query@0dB) ...", flush=True)
    cross = []
    for rep in range(3):
        rng_s = np.random.RandomState(9000 + rep)
        cs = rng_s.choice(all_classes, size=5, replace=False)
        smp_s = EpisodeSampler(zn, y, snr, classes=cs, n_way=5, k_shot=5,
                               q_per_class=15, seed=9090 + rep, snr_min=18, snr_max=18)
        smp_q = EpisodeSampler(zn, y, snr, classes=cs, n_way=5, k_shot=5,
                               q_per_class=15, seed=9500 + rep, snr_min=0, snr_max=0)
        Zs, _, _ = smp_s.sample(epi)
        _, Zq, yq = smp_q.sample(epi)   # 注意 sample 返回 (Zs, Zq, yq)
        Zs = Zs.astype(np.complex128); Zq = Zq.astype(np.complex128)
        row = {"rep": rep, "acc": {}}
        for mth in METHODS:
            if mth == "whiten":
                mu = E.proto_euclid(Zs); aux = E.whiten_aux(Zq)
                pred = E.cls_mahalanobis_diag(Zq, mu, aux)
            else:
                pred, _ = E.run_method(mth, Zs, Zq, mu_true=None, kappa_meta=np.pi / 3)
            row["acc"][mth] = float((pred == yq).mean())
        cross.append(row)
        print(f"  cross-SNR rep{rep}: " + " ".join(f"{m}={100*v:.1f}" for m, v in row["acc"].items()
                                                    if m in ("euclid", "orbital", "phasemap_ml")), flush=True)
    json.dump(cross, open(os.path.join(out, "logs", "B8_crosssnr.json"), "w"), indent=1)
    print(f"B8 DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
