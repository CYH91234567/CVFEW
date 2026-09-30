"""数据加载与基础工具（CVFEW 方向⑧）。"""
import numpy as np

MOD_CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4",
               "QAM16", "QAM64", "QPSK", "WBFM"]  # RML2016.10a 11类（缓存顺序待运行时核对）


def load_radioml(path):
    d = np.load(path)
    z, y, snr = d["z"], d["y"].astype(np.int64), d["snr"].astype(np.int64)
    return z, y, snr


def energy_normalize(z, eps=1e-12):
    """逐帧能量归一化（AGC式）。"""
    e = np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    return z / np.maximum(e, eps)


def features(z, mode):
    """复数帧 -> 实值特征。mode: amp | rip (Re,Im) | apc (amp+cos/sin phase)"""
    if mode == "amp":
        return np.abs(z)
    if mode == "rip":
        return np.concatenate([z.real, z.imag], axis=1)
    if mode == "apc":
        a = np.abs(z)
        ph = np.angle(z)
        return np.concatenate([a, np.cos(ph), np.sin(ph)], axis=1)
    raise ValueError(mode)


def randomize_global_phase(z, seed=0):
    """每帧独立随机全局相位旋转（信息删除型干预：仅删逐帧全局相位，保留帧内相对相位）。"""
    rng = np.random.RandomState(seed)
    u = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(z.shape[0], 1)))
    return (z * u).astype(np.complex64)


def softmax_train(Xtr, ytr, Xte, yte, K, lr=0.3, epochs=300, seed=0, wd=1e-4):
    """numpy 多类softmax线性探针（全批梯度下降+动量）。返回测试准确率。"""
    n, d = Xtr.shape
    rng = np.random.RandomState(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    W = np.zeros((d, K)); b = np.zeros(K)
    mW = np.zeros_like(W); mb = np.zeros_like(b)
    onehot = np.eye(K)[ytr]
    for ep in range(epochs):
        logits = Xtr_n @ W + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        g = (p - onehot) / n
        gW = Xtr_n.T @ g + wd * W
        gb = g.sum(0)
        mW = 0.9 * mW - lr * gW; mb = 0.9 * mb - lr * gb
        W += mW; b += mb
    acc = float((Xte_n @ W + b).argmax(1).__eq__(yte).mean())
    return acc
