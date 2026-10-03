"""B35 第一步：扩展不变统计量的信息含量探针（本地零训练，确定性，无 GPU/无 torch）。

目的：论文 B 的 limitation 指出"higher-order cyclic cumulants and wider lags are the
natural extension"；B33 显示 split-B（8PSK/CPFSK/QAM64）上 0/8 好盆 = 绝对阈值不迁移。
本探针回答一个先决问题：**扩展不变统计量是否在 split-B 难类上携带显著更多的类信息？**

协议（与项目估计器级实验同构，零训练）：
- 数据：本地 10b 缓存（RML2016.10b，无 AM-SSB；split-B 三类 {0:8PSK,3:CPFSK,7:QAM64}
  与 split-0 对照三类 {1:AM-DSB,6:QAM16,8:QPSK} 在 10b 编号下与 10a 一致）
- 3-way 5-shot，q=15/类，disjoint 支持/查询（复用 cvfe.episodes.EpisodeSampler）
- 4 个 (split, SNR) × 2 split × 4 SNR = 8 格 × 200 episode，全特征集共享同一 episode
  流（配对），逐 episode 计配对差
- 分类器：最近原型（标准化后均值原型 + 欧氏距离），标准化统计量取自 episode 内部
  （零训练；无任何学习成分 → 本地允许，与 local_b34_det 同级）

特征族（全部对 h→e^{iθ}h 精确不变——只含相位差与模的函数）：
- cur    : R̂(τ) τ=1..6（复）+ κ=m4/m2² + log m2        = nets_b19.invariant_stats（26 实维）
- wide   : R̂(τ) τ=1..16（复）+ κ + log m2              （66 实维）
- ho     : cur + 归一化四阶累积量 C40/m2²、C42/m2² + 二阶共轭无关量 C20/m2（30 实维）
- cyc    : cur + 循环自相关 R^α(0)、R^α(1) at α∈{1/16,1/8,1/4}（复）（26+12=38 实维）
- full   : wide + ho 的四阶量 + cyc 的循环量（78 实维）

判据（预注册式，探针级）：
- P35-a（信息增益）：full − cur 的配对精度差在 split-B 各 SNR 格 > 0 且 Wilcoxon p<0.05
- P35-b（量级）：split-B 上 full 相对 cur 的增益 ≥ 5pt（若 ≥5pt 则支持"B35 全量实验
  （学习 trunk + 扩展读出 + 度量头）可攻绝对阈值"的结论；若 <5pt 则不变统计的信息
  边界确实是瓶颈，论文 B limitation 的"natural extension"表述需削弱）
- P35-c（对照）：split-0（易类）上增益应小于 split-B（若反普适增益则说明扩展量主要
  在难类上起作用）

产物：04_results/logs/B35_probe.json（不覆盖任何既有文件）
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe.episodes import EpisodeSampler  # noqa: E402

CACHE = str(Path(__file__).resolve().parents[1] / "04_results" / "local_b10b" / "cache_b10b.npz")
OUT = str(Path(__file__).resolve().parents[1] / "04_results" / "logs" / "B35_probe.json")

SPLIT_B = [0, 3, 7]     # 8PSK, CPFSK, QAM64（10b 编号与 10a 一致）
SPLIT_0 = [1, 6, 8]     # AM-DSB, QAM16, QPSK
SNRS = [0, 6, 12, 18]
N_EP = 200
TAU_CUR = 6
TAU_WIDE = 16
CYC_ALPHAS = [1.0 / 16, 1.0 / 8, 1.0 / 4]


def feats_cur(h):
    """h:(B,L) complex -> (B,D) float。与 nets_b19.invariant_stats 同构（一帧一通道）。"""
    L = h.shape[-1]
    a2 = np.abs(h) ** 2
    m2 = a2.mean(-1)
    m2 = np.maximum(m2, 1e-12)
    out = []
    for tau in range(1, TAU_CUR + 1):
        r = (h[..., tau:] * np.conj(h[..., : L - tau])).mean(-1) / m2
        out.append(r.real)
        out.append(r.imag)
    kap = (a2 ** 2).mean(-1) / m2 ** 2
    out.append(kap)
    out.append(np.log(m2))
    return np.stack(out, axis=-1)


def feats_wide(h):
    L = h.shape[-1]
    a2 = np.abs(h) ** 2
    m2 = np.maximum(a2.mean(-1), 1e-12)
    out = []
    for tau in range(1, TAU_WIDE + 1):
        r = (h[..., tau:] * np.conj(h[..., : L - tau])).mean(-1) / m2
        out.append(r.real)
        out.append(r.imag)
    out.append((a2 ** 2).mean(-1) / m2 ** 2)
    out.append(np.log(m2))
    return np.stack(out, axis=-1)


def _ho_extra(h):
    """归一化四阶累积量 + 二阶共轭无关量（全部相位不变）。"""
    a2 = np.abs(h) ** 2
    m2 = np.maximum(a2.mean(-1), 1e-12)
    m2sq = m2 ** 2
    c40 = (h ** 4).mean(-1) / m2sq            # E[h^4]/m2^2（复：8PSK 4 次幂→2 点）
    c42 = ((a2 ** 2).mean(-1) - 2 * m2sq) / m2sq   # E[|h|^4]-2m2^2 / m2^2（实）
    c20 = (h ** 2).mean(-1) / m2              # E[h^2]/m2（复：QAM 方格结构）
    return np.stack([c40.real, c40.imag, c42, c20.real, c20.imag], axis=-1)


def _cyc_extra(h):
    """循环自相关 R^α(τ) = mean_t h(t+τ) conj(h(t)) e^{-i2παt}（相位不变）。"""
    L = h.shape[-1]
    a2 = np.abs(h) ** 2
    m2 = np.maximum(a2.mean(-1), 1e-12)
    out = []
    t = np.arange(L)
    for tau in (0, 1):
        ht = np.conj(h[..., : L - tau]) if tau else np.conj(h)
        hs = h[..., tau:]
        tw = t[: hs.shape[-1]]
        for a in CYC_ALPHAS:
            r = (hs * ht * np.exp(-1j * 2 * np.pi * a * tw)).mean(-1) / m2
            out.append(r.real)
            out.append(r.imag)
    return np.stack(out, axis=-1)


FEATS = {
    "cur": feats_cur,
    "wide": feats_wide,
    "ho": lambda h: np.concatenate([feats_cur(h), _ho_extra(h)], axis=-1),
    "cyc": lambda h: np.concatenate([feats_cur(h), _cyc_extra(h)], axis=-1),
    "full": lambda h: np.concatenate(
        [feats_wide(h), _ho_extra(h), _cyc_extra(h)], axis=-1),
}


def classify_episode(Fs, Fq, n_way, k_shot, q):
    """零训练最近原型：episode 内标准化（支持+查询合并估计均值方差），支持均值原型。"""
    allf = np.concatenate([Fs.reshape(-1, Fs.shape[-1]), Fq], axis=0)
    mu = allf.mean(0)
    sd = allf.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Fs_n = (Fs.reshape(n_way, k_shot, -1) - mu) / sd
    Fq_n = (Fq - mu) / sd
    proto = Fs_n.mean(1)                                   # (n_way, D)
    d = ((Fq_n[:, None, :] - proto[None, :, :]) ** 2).sum(-1)   # (n_way*q, n_way)
    return d.argmin(1)


def run_cell(z, y, snr_arr, classes, snr_val, seed):
    sampler = EpisodeSampler(z, y, snr_arr, classes, n_way=3, k_shot=5,
                             q_per_class=15, seed=seed)
    Zs, Zq, yq = sampler.sample(N_EP, query_snr=snr_val, energy_norm=True,
                                disjoint=True)
    accs = {name: np.zeros(N_EP) for name in FEATS}
    for e in range(N_EP):
        # 支持/查询能量归一化（与 episodes.py 一致，在特征前做）
        zs = Zs[e] / np.maximum(np.abs(Zs[e]).std(), 1e-12)
        zq = Zq[e] / np.maximum(np.abs(Zq[e]).std(), 1e-12)
        Fs = {n: f(zs) for n, f in FEATS.items()}
        Fq = {n: f(zq) for n, f in FEATS.items()}
        n_way, k, q = 3, 5, 15
        for n in FEATS:
            pred = classify_episode(Fs[n], Fq[n], n_way, k, q)
            accs[n][e] = (pred == yq[e]).mean()
    return accs


def main():
    d = np.load(CACHE, allow_pickle=True)
    z, y, snr_arr = d["z"], d["y"], d["snr"]
    print(f"[b35] cache z={z.shape} classes={np.unique(y)} snrs={np.unique(snr_arr)}")
    results = []
    for split_name, classes in (("split0", SPLIT_0), ("splitB", SPLIT_B)):
        for sv in SNRS:
            seed = 20261003 + 1000 * sv + (7 if split_name == "splitB" else 3)
            accs = run_cell(z, y, snr_arr, classes, sv, seed)
            row = {"split": split_name, "classes": classes, "snr": sv,
                   "n_episodes": N_EP}
            for n, a in accs.items():
                row[f"{n}_mean"] = float(a.mean())
                row[f"{n}_sd"] = float(a.std())
            # 配对差 full - cur（及其它对比）
            for n in ("wide", "ho", "cyc", "full"):
                dd = accs[n] - accs["cur"]
                p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                row[f"{n}_minus_cur_mean"] = float(dd.mean())
                row[f"{n}_minus_cur_p"] = float(p)
                row[f"{n}_minus_cur_win"] = float((dd > 0).mean())
            results.append(row)
            print(f"[b35] {split_name} SNR={sv:>2}: " + " ".join(
                f"{n}={row[n+'_mean']*100:.2f}" for n in ("cur", "wide", "ho", "cyc", "full"))
                + f" | full-cur={row['full_minus_cur_mean']*100:+.2f}pt p={row['full_minus_cur_p']:.2e}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[b35] written {OUT}")

    # 判据汇总
    sb = [r for r in results if r["split"] == "splitB"]
    s0 = [r for r in results if r["split"] == "split0"]
    print("\n=== B35 探针判据 ===")
    for r in sb:
        print(f"splitB SNR={r['snr']:>2}: full-cur = {r['full_minus_cur_mean']*100:+.2f}pt "
              f"(p={r['full_minus_cur_p']:.1e}, win={r['full_minus_cur_win']:.2f})")
    gain_sb = np.mean([r["full_minus_cur_mean"] for r in sb]) * 100
    gain_s0 = np.mean([r["full_minus_cur_mean"] for r in s0]) * 100
    print(f"P35-a (split-B 各格正向): "
          f"{all(r['full_minus_cur_mean'] > 0 for r in sb)}")
    print(f"P35-b (split-B 均增益 ≥5pt): {gain_sb:+.2f}pt -> "
          f"{'PASS' if gain_sb >= 5 else 'FAIL'}")
    print(f"P35-c (split-0 增益 < split-B): {gain_s0:+.2f} vs {gain_sb:+.2f} -> "
          f"{'PASS' if gain_s0 < gain_sb else 'FAIL'}")


if __name__ == "__main__":
    main()
