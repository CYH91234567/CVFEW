"""B1 诊断：真实IQ的自然相位结构 + 商前提三值门。

D1 自然逐帧全局相位相干度 R̄（类×SNR）：同/跨类帧对内积相位的加权合向量长度。
   若数据集逐帧全局相位随机（接收机相位随机），R̄≈0；若帧间锁相/类条件全局相位，R̄_same 显著>0。
D2 帧内相对相位结构：相邻样本相位差 Δφ_k = arg(z_{k+1} conj(z_k)) 的均值（CFO斜率）与标准差（相位噪声）。
D3 商前提探针：随机化每帧全局相位前后，线性softmax探针准确率差（类×SNR层）。
   - 随机化仅删除逐帧全局相位信息（每帧一个随机相位旋转，帧内相对相位结构不变）。
D4 幅度结构：帧能量分布、衰落尾部证据（异方差动机）。
D5 类×SNR 层的幅度-only vs 相位特征探针差（线性探针口径的相位信息代理）。

三值门（逻辑链 §2 H1a）：
  G1a 商前提成立     : D3 全局下降<1pt 且 R̄_same≈R̄_cross
  G1b 部分相干       : 中间情形（下降1-3pt 或 R̄_same 显著>0 但类条件不主导）
  G1c 绝对相位判别   : 全局下降>3pt 且 类均值相位分离显著 -> 方向级止损(R3)
"""
import json, os
import numpy as np
from . import data as D


def resultant_phase_table(z, y, snr, n_pairs=4000, seed=0, n_snr_bins=5):
    """R̄表：同/跨类帧对内积相位的加权/未加权合向量长度，按类×SNR档。
    返回 {class: {snr_bin: {'R_same_w','R_cross_w','R_same','R_cross','n'}}}
    """
    rng = np.random.RandomState(seed)
    snr_edges = np.quantile(snr, np.linspace(0, 1, n_snr_bins + 1))
    snr_edges[0] -= 1; snr_edges[-1] += 1
    out = {}
    classes = np.unique(y)
    for c in classes:
        out[int(c)] = {}
        for b in range(n_snr_bins):
            m = (y == c) & (snr >= snr_edges[b]) & (snr < snr_edges[b + 1])
            idx = np.where(m)[0]
            if len(idx) < 40:
                out[int(c)][b] = None
                continue
            i = rng.choice(idx, size=min(len(idx), 200), replace=False)
            Zc = z[i]
            # 同类对
            rs_w, rs, ntot = _pair_phase_R(Zc, Zc, n_pairs, rng, self_pair=True)
            # 跨类对
            others = classes[classes != c]
            Zx = z[rng.choice(np.where(np.isin(y, others) & (snr >= snr_edges[b]) & (snr < snr_edges[b + 1]))[0],
                              size=min(200, 4000), replace=False)] if len(others) else None
            if Zx is None or len(Zx) < 20:
                out[int(c)][b] = None
                continue
            rc_w, rc, _ = _pair_phase_R(Zc, Zx, n_pairs, rng, self_pair=False)
            out[int(c)][b] = {"R_same_w": rs_w, "R_cross_w": rc_w,
                              "R_same": rs, "R_cross": rc, "n": int(ntot)}
    return out


def _pair_phase_R(A, B, n_pairs, rng, self_pair):
    """帧对内积相位合向量长度。加权权=|<a,b>|。A:(nA,L) B:(nB,L)"""
    nA, nB = len(A), len(B)
    if self_pair:
        # 无放近近似：随机 i<j 对
        pairs = set()
        while len(pairs) < min(n_pairs, nA * (nA - 1) // 2):
            i, j = rng.randint(0, nA, 2)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
        pairs = list(pairs)
    else:
        pairs = [(rng.randint(0, nA), rng.randint(0, nB)) for _ in range(min(n_pairs, nA * nB))]
    if not pairs:
        return 0.0, 0.0, 0
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    ip = np.sum(np.conj(A[ii]) * B[jj], axis=1)          # 内积
    w = np.abs(ip)
    ph = np.angle(ip)
    R_w = float(np.abs(np.sum(w * np.exp(1j * ph))) / max(np.sum(w), 1e-12))
    R = float(np.abs(np.sum(np.exp(1j * ph))) / len(pairs))
    return R_w, R, len(pairs)


def intraframe_phase_stats(z, y, snr, n_snr_bins=5, seed=0):
    """帧内相邻样本相位差统计（CFO斜率与相位噪声），类×SNR档。"""
    snr_edges = np.quantile(snr, np.linspace(0, 1, n_snr_bins + 1))
    snr_edges[0] -= 1; snr_edges[-1] += 1
    out = {}
    dphi = np.angle(z[:, 1:] * np.conj(z[:, :-1]))       # (n, L-1)
    for c in np.unique(y):
        out[int(c)] = {}
        for b in range(n_snr_bins):
            m = (y == c) & (snr >= snr_edges[b]) & (snr < snr_edges[b + 1])
            if m.sum() < 40:
                out[int(c)][b] = None
                continue
            dp = dphi[m]
            Rm = np.abs(np.mean(np.exp(1j * dp)))         # 相位差集中度
            out[int(c)][b] = {"dphi_mean": float(np.mean(dp)),
                              "dphi_std": float(np.std(dp)),
                              "dphi_R": float(Rm), "n": int(m.sum())}
    return out


def randomization_probe(z, y, snr, out_json, seed=0, n_train=20000, n_test=8000,
                        n_snr_bins=5, epochs=300):
    """D3/D5：商前提探针（匹配分布重训协议）+ 特征模态差。

    纪律（路线文档 §1.2(2)/§3.3）：直接"train自然→test随机化"会制造分布外输入，
    性能下降不构成信息删除结论。正确协议：
      acc_nat  = train(自然帧) -> test(自然帧)
      acc_rand = train(重随机化帧,seedα) -> test(重随机化帧,seedβ≠α)
    drop_matched = acc_nat - acc_rand：两边分布各自内部一致，差值才是
    "逐帧全局相位携带的标签信息"（线性探针口径）。
    另报 OOD drop（train自然→test随机化）仅作为旋转敏感性描述，不用于判门。
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    tr, te = perm[:n_train], perm[n_train:n_train + n_test]
    zr_a = D.randomize_global_phase(z, seed=seed + 1)
    zr_b = D.randomize_global_phase(z, seed=seed + 2)

    snr_edges = np.quantile(snr, np.linspace(0, 1, n_snr_bins + 1))
    snr_edges[0] -= 1; snr_edges[-1] += 1
    res = {"per_feature": {}, "per_snr": [], "per_class": {}}
    K = int(y.max()) + 1

    for mode in ["rip", "amp", "apc"]:
        X = D.features(z, mode)
        Xa = D.features(zr_a, mode)
        Xb = D.features(zr_b, mode)
        acc_nat = D.softmax_train(X[tr], y[tr], X[te], y[te], K, epochs=epochs, seed=seed)
        acc_rand = D.softmax_train(Xa[tr], y[tr], Xb[te], y[te], K, epochs=epochs, seed=seed)
        acc_ood = D.softmax_train(X[tr], y[tr], Xa[te], y[te], K, epochs=epochs, seed=seed)
        res["per_feature"][mode] = {"acc": acc_nat, "acc_randphase_matched": acc_rand,
                                    "drop_matched": acc_nat - acc_rand,
                                    "acc_ood_testonly": acc_ood,
                                    "drop_ood": acc_nat - acc_ood}
        print(f"  [probe:{mode}] nat={acc_nat:.4f} rand_matched={acc_rand:.4f} "
              f"drop_matched={acc_nat - acc_rand:+.4f} (ood={acc_nat - acc_ood:+.4f})", flush=True)

    # rip 模态按 SNR 档（匹配分布）
    X = D.features(z, "rip"); Xa = D.features(zr_a, "rip"); Xb = D.features(zr_b, "rip")
    for b in range(n_snr_bins):
        mtr = (snr[tr] >= snr_edges[b]) & (snr[tr] < snr_edges[b + 1])
        mte = (snr[te] >= snr_edges[b]) & (snr[te] < snr_edges[b + 1])
        if mtr.sum() < 500 or mte.sum() < 300:
            continue
        acc_nat = D.softmax_train(X[tr][mtr], y[tr][mtr], X[te][mte], y[te][mte], K, epochs=epochs, seed=seed)
        acc_rand = D.softmax_train(Xa[tr][mtr], y[tr][mtr], Xb[te][mte], y[te][mte], K, epochs=epochs, seed=seed)
        res["per_snr"].append({"snr_bin": b, "lo": float(snr_edges[b]), "hi": float(snr_edges[b + 1]),
                               "acc": acc_nat, "acc_randphase_matched": acc_rand,
                               "drop_matched": acc_nat - acc_rand})
        print(f"  [probe snr {snr_edges[b]:.0f},{snr_edges[b+1]:.0f}] drop_matched={acc_nat - acc_rand:+.4f}", flush=True)

    # rip 模态按类（匹配分布，one-vs-rest 平衡采样）
    rng_c = np.random.RandomState(seed + 7)
    for c in np.unique(y):
        pos_tr = np.where(y[tr] == c)[0]; pos_te = np.where(y[te] == c)[0]
        neg_tr = np.where(y[tr] != c)[0]; neg_te = np.where(y[te] != c)[0]
        neg_tr_s = rng_c.choice(neg_tr, size=len(pos_tr), replace=False)
        neg_te_s = rng_c.choice(neg_te, size=len(pos_te), replace=False)
        itr = np.concatenate([pos_tr, neg_tr_s]); ite = np.concatenate([pos_te, neg_te_s])
        ytr_bin = (y[tr][itr] == c).astype(np.int64); yte_bin = (y[te][ite] == c).astype(np.int64)
        acc_nat = D.softmax_train(X[tr][itr], ytr_bin, X[te][ite], yte_bin, 2, epochs=epochs, seed=seed)
        acc_rand = D.softmax_train(Xa[tr][itr], ytr_bin, Xb[te][ite], yte_bin, 2, epochs=epochs, seed=seed)
        res["per_class"][int(c)] = {"acc": acc_nat, "acc_randphase_matched": acc_rand,
                                    "drop_matched": acc_nat - acc_rand}
    return res


def carrier_reference_diag(z, y, seed=0):
    """DC分量诊断：逐帧均值（DC/载波泄漏分量）的幅度占比与跨帧相位集中度（按类）。
    若某些类 |E[z]|/RMS 高且 DC 相位跨帧集中 -> 数据集存在固定相位参考约定（载体分量），
    全局相位信息主要寄居于此 -> 属性为'接收机参考相位'而非类语义。"""
    dc = z.mean(axis=1)                                  # (n,) 复数
    e = np.sqrt(np.mean(np.abs(z) ** 2, axis=1)) + 1e-12
    rel = np.abs(dc) / e
    out = {"per_class": {}}
    for c in np.unique(y):
        m = y == c
        dcc = dc[m]
        R = np.abs(np.mean(dcc / (np.abs(dcc) + 1e-12)))  # DC相位跨帧集中度
        out["per_class"][int(c)] = {"dc_rel_mean": float(rel[m].mean()),
                                    "dc_rel_p90": float(np.quantile(rel[m], 0.9)),
                                    "dc_phase_R": float(R), "n": int(m.sum())}
    return out


def amplitude_stats(z, y, snr, n_snr_bins=5):
    e = np.mean(np.abs(z) ** 2, axis=1)
    out = {"energy_overall": {"mean": float(e.mean()), "std": float(e.std()),
                              "cv": float(e.std() / max(e.mean(), 1e-12)),
                              "p95_p50": float(np.quantile(e, 0.95) / max(np.quantile(e, 0.5), 1e-12))},
           "per_class": {}}
    snr_edges = np.quantile(snr, np.linspace(0, 1, n_snr_bins + 1))
    snr_edges[0] -= 1; snr_edges[-1] += 1
    # 帧内坐标幅度方差（异方差证据：同帧内不同坐标幅度波动）
    for c in np.unique(y):
        m = y == c
        percoord = np.std(np.abs(z[m]), axis=0)
        out["per_class"][int(c)] = {"energy_mean": float(e[m].mean()),
                                    "percoord_cv": float(np.std(percoord) / max(np.mean(percoord), 1e-12)),
                                    "n": int(m.sum())}
    return out


def classify_gate(probe_res, rbar_table, drop_thr=(1.0, 3.0)):
    """三值门判定（匹配分布口径）。G1c另需类均值相位分离证据（R̄比）。"""
    g = probe_res["per_feature"]["rip"]
    drop = 100 * g["drop_matched"]
    same, cross = [], []
    for c, bins in rbar_table.items():
        for b, rec in bins.items():
            if rec:
                same.append(rec["R_same_w"]); cross.append(rec["R_cross_w"])
    R_same, R_cross = float(np.mean(same)), float(np.mean(cross))
    if drop > drop_thr[1]:
        gate = "G1c"
    elif drop < drop_thr[0] and R_same < 0.2:
        gate = "G1a"
    else:
        gate = "G1b"
    return {"gate": gate, "rip_drop_matched_pts": drop,
            "rip_drop_ood_pts": 100 * g["drop_ood"],
            "R_same_w_mean": R_same, "R_cross_w_mean": R_cross,
            "amp_drop_matched_pts": 100 * probe_res["per_feature"]["amp"]["drop_matched"],
            "apc_drop_matched_pts": 100 * probe_res["per_feature"]["apc"]["drop_matched"]}


def run_b1(cache_path, out_dir, seed=0):
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    z, y, snr = D.load_radioml(cache_path)
    print(f"data: z{z.shape} {z.dtype} classes={np.unique(y)} snr[{snr.min()},{snr.max()}]", flush=True)
    zn = D.energy_normalize(z)

    print("[D1] resultant phase table ...", flush=True)
    rbar = resultant_phase_table(zn, y, snr, seed=seed)
    print("[D2] intraframe phase stats ...", flush=True)
    intra = intraframe_phase_stats(zn, y, snr)
    print("[D3/D5] randomization probe ...", flush=True)
    probe = randomization_probe(zn, y, snr, None, seed=seed)
    print("[D4] amplitude stats ...", flush=True)
    amp = amplitude_stats(zn, y, snr)
    print("[D6] DC carrier reference diag ...", flush=True)
    carrier = carrier_reference_diag(zn, y, seed=seed)

    gate = classify_gate(probe, rbar)
    print(f"\n=== GATE: {gate['gate']} (rip drop_matched {gate['rip_drop_matched_pts']:.2f} pts, "
          f"R_same={gate['R_same_w_mean']:.3f}, R_cross={gate['R_cross_w_mean']:.3f}) ===", flush=True)

    res = {"gate": gate, "probe": probe, "amplitude": amp, "carrier": carrier}
    json.dump(res, open(os.path.join(out_dir, "logs", "B1_diagnostic.json"), "w"), indent=2)
    json.dump({"rbar": rbar, "intraframe": intra},
              open(os.path.join(out_dir, "logs", "B1_phase_structure.json"), "w"), indent=2)
    return res
