"""RadioML RML2016.10a 少样本 episode 采样器（B8）。

协议（逻辑链 §5.4）：
  - 类划分：11 类 -> meta-train/val/test = 6/2/3，5 个随机划分（分层保持模拟/数字类平衡）
  - episode：N-way K-shot（N=5），support 与 query 取自同一 SNR 层（防 SNR 泄漏）；
    跨 SNR 层（高SNR support -> 低SNR query）作为难迁移协议单独报告
  - 相位注入三件套（叠加在自然损伤之上，定位为导频辅助接收场景）：
    global: 每帧随机全局相位 e^{iφ}, φ~wrapN(0, σ_θ²)
    pnoise: 帧内 Wiener 相位噪声（随机游走）
    cfo:    线性相位斜坡 e^{i2πf_0 t}（U(1)^p，检验不误伤相对相位）
  - 类别索引约定：缓存 y ∈ {0..10}，调制名映射见 MOD_CLASSES
"""
import numpy as np

MOD_CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4",
               "QAM16", "QAM64", "QPSK", "WBFM"]
DIGITAL = {"8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK"}


def make_class_splits(n_classes=11, n_test=3, n_val=2, n_splits=5, seed=20260929):
    """分层随机划分：每划分保持数字/模拟类比例。返回 [ {'train':[...],'val':[...],'test':[...]} ]"""
    rng = np.random.RandomState(seed)
    dig = [c for c in range(n_classes) if MOD_CLASSES[c] in DIGITAL]
    ana = [c for c in range(n_classes) if MOD_CLASSES[c] not in DIGITAL]
    splits = []
    for _ in range(n_splits):
        d = rng.permutation(dig).tolist()
        a = rng.permutation(ana).tolist()
        pool = d + a
        rng.shuffle(pool)
        test = pool[:n_test]
        val = pool[n_test:n_test + n_val]
        train = pool[n_test + n_val:]
        splits.append({"train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return splits


def inject_phase(z, mode, strength, rng):
    """z:(n,L) complex。mode: none|global|pnoise|cfo。strength: σ_θ / 扩散系数 / f_0(周期/帧)"""
    n, L = z.shape
    if mode == "global":
        phi = strength * rng.randn(n, 1)
        return z * np.exp(1j * phi)
    if mode == "pnoise":
        steps = strength * rng.randn(n, L)
        walk = np.cumsum(steps, axis=1)
        return z * np.exp(1j * walk)
    if mode == "cfo":
        t = np.arange(L)[None, :] / L
        f0 = strength * rng.randn(n, 1)
        return z * np.exp(1j * 2 * np.pi * f0 * t)
    return z


class EpisodeSampler:
    """按类划分与 SNR 层采样 N-way K-shot episode。query_per_class 默认 15。"""

    def __init__(self, z, y, snr, classes, n_way=5, k_shot=5, q_per_class=15,
                 snr_min=None, snr_max=None, seed=0):
        self.z, self.y, self.snr = z, y, snr
        self.classes = list(classes)
        self.n_way, self.k_shot, self.q = n_way, k_shot, q_per_class
        self.rng = np.random.RandomState(seed)
        self.idx_by = {}
        for c in self.classes:
            for s in (np.unique(snr) if (snr_min is None) else [s for s in np.unique(snr)
                                                               if snr_min <= s <= snr_max]):
                m = np.where((y == c) & (snr == s))[0]
                if len(m) >= k_shot + q_per_class:
                    self.idx_by[(c, int(s))] = m

    def __len__(self):
        return 10 ** 9

    def sample(self, n_episodes, query_snr=None, inject=None, inj_strength=0.0,
               energy_norm=True):
        """返回 Zs:(E,N,k,L), Zq:(E,N*q,L), yq:(E,N*q)。query_snr=None 时同层。
        inject: (mode, strength) 或 None。注意注入在采样后叠加（叠加自然损伤之上）。"""
        E = n_episodes
        N, k, q, L = self.n_way, self.k_shot, self.q, self.z.shape[1]
        Zs = np.zeros((E, N, k, L), dtype=np.complex64)
        Zq = np.zeros((E, N * q, L), dtype=np.complex64)
        yq = np.zeros((E, N * q), dtype=np.int64)
        all_snrs = sorted({s for (c, s) in self.idx_by})
        for e in range(E):
            cs = self.rng.choice(self.classes, size=N, replace=False)
            s_ref = all_snrs[self.rng.randint(len(all_snrs))]
            for i, c in enumerate(cs):
                s_sup = s_ref
                s_qr = s_ref if query_snr is None else query_snr
                pool_s = self.idx_by.get((c, int(s_sup)))
                pool_q = self.idx_by.get((c, int(s_qr)))
                tries = 0
                while (pool_s is None or len(pool_s) < k or pool_q is None or len(pool_q) < q) and tries < 20:
                    s_sup = all_snrs[self.rng.randint(len(all_snrs))]
                    s_qr = s_sup if query_snr is None else all_snrs[self.rng.randint(len(all_snrs))]
                    pool_s = self.idx_by.get((c, int(s_sup)))
                    pool_q = self.idx_by.get((c, int(s_qr)))
                    tries += 1
                if pool_s is None or pool_q is None:
                    raise RuntimeError("no pool for class/snr")
                si = self.rng.choice(pool_s, size=k, replace=False)
                qi = self.rng.choice(pool_q, size=q, replace=False)
                Zs[e, i] = self.z[si]
                Zq[e, i * q:(i + 1) * q] = self.z[qi]
                yq[e, i * q:(i + 1) * q] = i
        if inject is not None and inject[0] != "none":
            mode, strength = inject
            shape_s = Zs.shape
            Zs = inject_phase(Zs.reshape(-1, L), mode, strength, self.rng).reshape(shape_s)
            Zq = inject_phase(Zq.reshape(-1, L), mode, strength, self.rng).reshape(Zq.shape)
        if energy_norm:
            en = np.sqrt(np.mean(np.abs(Zs) ** 2, axis=-1, keepdims=True))
            Zs = Zs / np.maximum(en, 1e-12)
            enq = np.sqrt(np.mean(np.abs(Zq) ** 2, axis=-1, keepdims=True))
            Zq = Zq / np.maximum(enq, 1e-12)
        return Zs, Zq, yq
