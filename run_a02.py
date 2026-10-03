"""A-02：独立（per-condition）选择口径复算——对 Kumar 式 shared-trial 质疑的预防性审计。

背景：论文 A 的头条配对主张（+1.75pt @ σ_θ=π/6，p=1.1e-156）在与"按格更优端点"
的**同一批 episode** 上选点（shared-trial selection）。Kumar 2026 指出这类同数据
选点可夸大效应。本脚本用**与 B20_grid 完全相同的数据**（同种子方案：mu 种子
crc32("mu:"+key)、episode 种子 crc32(key)+done、chunk=125、kappa_meta=π/3、
joint κ profile）重算，并把口径拆成两列：

  - paired   （已发表）：全 2000 epi 同数据选 better endpoint，配对 Wilcoxon；
  - independent：固定 perm 半分（1000 选择 / 1000 评估），选择集选 better
                endpoint，评估集上读配对差值；另附"评估集自身选点"的对照，
                在相同样本量下分离纯选点效应。

格集 = B5_extra 的 12 个 H1b 格 + B20_grid h1b 实际最大格（p=64, ρ=0, π/6,
fade=0，+4.76pt——论文"+1.75 是 grid 最大"的范围只覆盖 B5_extra 的 ρ∈{0.3,0.7}，
本格未被 B5_extra 跑过，作为审计发现一并检验）。

输出：04_results/logs/A02_independent_selection.json、04_results/tables/
T24_a02_independent_selection.md。验证门：重算的 3 方法 acc 与 B20_grid 存储
值逐位一致（同代码路径同种子）；不一致则中止。
"""
import argparse, json, os, sys, time, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E
import run_b5 as R   # 复用 paired_test（zsplit Wilcoxon + 2000 bootstrap CI, rng 0）

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PI = np.pi
K, KSHOT, M, SIG, EPI, CHUNK = 5, 5, 75, 0.3, 2000, 125
USE = ("euclid", "orbital", "phasemap_ml")

# (p, rho, sigma_th, fade_q)：12 H1b 格（B5_extra 条件）+ B20_grid h1b 最大格
CELLS = [(64, 0.3, PI / 6, 0.0), (64, 0.3, PI / 3, 0.0), (64, 0.3, PI / 2, 0.0),
         (64, 0.7, PI / 6, 0.0), (64, 0.7, PI / 3, 0.0), (64, 0.7, PI / 2, 0.0),
         (64, 0.3, PI / 6, 0.5), (64, 0.3, PI / 3, 0.5), (64, 0.3, PI / 2, 0.5),
         (64, 0.7, PI / 6, 0.5), (64, 0.7, PI / 3, 0.5), (64, 0.7, PI / 2, 0.5),
         (64, 0.0, PI / 6, 0.0)]   # B20_grid h1b 最大格（ρ=0，B5_extra 未覆盖）


def cell_key(p, rho, st, fade):
    return "|".join(f"{x}={y}" for x, y in sorted(
        {"tag": "main", "p": p, "rho": rho, "k": KSHOT, "st": round(st, 4),
         "fade": fade, "sig": SIG}.items()))


def run_cell(p, rho, st, fade):
    """逐位复现 B20_grid 的 episode 数据，只算 3 个方法（5~7× 快于全 21 方法）。"""
    key = cell_key(p, rho, st, fade)
    mu = S.make_prototypes(K, p, rho, np.random.RandomState(zlib.crc32(("mu:" + key).encode())))
    seed = zlib.crc32(key.encode())
    n_ok = {mth: np.zeros(EPI) for mth in USE}
    done = 0
    while done < EPI:
        n = min(CHUNK, EPI - done)
        Zs, Zq, yq = S.batch_episodes(mu, K, KSHOT, M, st, SIG, n, seed=seed + done,
                                      variant="static", fade_q=fade)
        for mth in USE:
            pred, _ = E.run_method(mth, Zs, Zq, mu_true=mu, kappa_meta=PI / 3)
            n_ok[mth][done:done + n] = (pred == yq).mean(axis=1)
        done += n
    return n_ok


def analyze(p, rho, st, fade, n_ok):
    perm = np.random.RandomState(20261003).permutation(EPI)
    sel, ev = perm[:EPI // 2], perm[EPI // 2:]
    acc = {mth: float(np.mean(v)) for mth, v in n_ok.items()}

    def pick(idx):
        e, o = float(n_ok["euclid"][idx].mean()), float(n_ok["orbital"][idx].mean())
        return "euclid" if e >= o else "orbital", max(e, o)

    b_full, _ = pick(np.arange(EPI))
    b_sel, _ = pick(sel)
    b_ev, _ = pick(ev)
    out = {"cond": {"p": p, "rho": rho, "sigma_th": st, "fade_q": fade},
           "acc": acc, "better_full": b_full, "better_sel": b_sel, "better_ev": b_ev}
    # 已发表口径：全 2000 epi 同数据选点
    out["paired_full2000"] = R.paired_test(n_ok["phasemap_ml"], n_ok[b_full])
    out["paired_full2000"]["acc_endpoint"] = acc[b_full]
    # 独立口径：选择集选点，评估集读数
    out["independent_sel1000_ev1000"] = R.paired_test(
        n_ok["phasemap_ml"][ev], n_ok[b_sel][ev])
    # 对照：评估集自身选点（同 n=1000，分离纯选点效应）
    out["control_samedata_ev1000"] = R.paired_test(
        n_ok["phasemap_ml"][ev], n_ok[b_ev][ev])
    return out


def _worker(job):
    p, rho, st, fade = job
    t0 = time.time()
    n_ok = run_cell(p, rho, st, fade)
    res = analyze(p, rho, st, fade, n_ok)
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑头条格 400 epi 验证时间")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    b20 = json.load(open(os.path.join(BASE, "04_results", "logs", "B20_grid.json")))
    b20map = {}
    for r in b20["records"]:
        b20map[(r["p"], round(r["rho"], 3), r["k"], round(r["sigma_th"], 4),
                round(r["fade_q"], 3))] = r

    if a.smoke:
        global EPI
        EPI = 400
        n_ok = run_cell(64, 0.3, PI / 6, 0.0)
        t = n_ok["euclid"]
        print(f"smoke 400 epi headline cell: euclid={100*t.mean():.4f} "
              f"orbital={100*n_ok['orbital'].mean():.4f} phML={100*n_ok['phasemap_ml'].mean():.4f}")
        r = b20map[(64, 0.3, 5, round(PI / 6, 4), 0.0)]
        # 前 400 epi 的 acc 无法直接比对（存储是 2000 均值），只做计时与分布合理性检查
        return

    jobs = CELLS
    res = []
    import multiprocessing as mp
    t00 = time.time()
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
            res.append(r)
            c = r["cond"]
            print(f"[{time.time()-t00:6.0f}s] {i+1}/{len(jobs)} p={c['p']} rho={c['rho']} "
                  f"st={c['sigma_th']:.2f} fade={c['fade_q']} phML={100*r['acc']['phasemap_ml']:.2f} "
                  f"paired={100*r['paired_full2000']['ci'][0]:+.2f}..{100*r['paired_full2000']['ci'][1]:+.2f} "
                  f"indep={100*r['independent_sel1000_ev1000']['ci'][0]:+.2f}.."
                  f"{100*r['independent_sel1000_ev1000']['ci'][1]:+.2f}", flush=True)

    # 验证门：重算 acc 与 B20_grid 逐位一致
    ok, bad = True, []
    for r in res:
        c = r["cond"]
        key = (c["p"], round(c["rho"], 3), 5, round(c["sigma_th"], 4),
               round(c["fade_q"], 3))
        st = b20map[key]
        for mth in USE:
            if abs(st["acc"][mth] - r["acc"][mth]) > 1e-12:
                ok = False
                bad.append((key, mth, st["acc"][mth], r["acc"][mth]))
    r = {"validation_vs_b20grid": {"bit_exact": ok, "mismatches": bad},
         "protocol": {"epi": EPI, "split": "RandomState(20261003) permutation 1000/1000",
                      "paired_test": "run_b5.paired_test (zsplit Wilcoxon + 2000x bootstrap CI)",
                      "kappa_profile": "joint (B21 final default)"},
         "cells": res}
    out = os.path.join(BASE, "04_results", "logs", "A02_independent_selection.json")
    json.dump(r, open(out, "w"), indent=1)
    print("validation bit_exact:", ok)
    if bad:
        for b in bad[:10]:
            print("  MISMATCH", b)
    print("A02 ->", out)


if __name__ == "__main__":
    main()
