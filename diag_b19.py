"""B19 诊断 D1-D3：随机初始化 ComplexAMC 的池化相消（本地 numpy，无需训练）。

H-a（池化相消屏障）：随机宽频卷积核 → h_c(t) 相位时间近均匀 → mean_t 相消 →
池化嵌入无信号 → 梯度纯噪声 → loss 停在 ln5。

D1 相消比 ρ_c = |mean_t h_c| / mean_t|h_c|（末层隐层，逐通道）
D2 均值池化嵌入的 Fisher 判别比 + orbital 头 5-way 精度（≈chance 预测）
D3 功率池化嵌入的 Fisher 判别比 + euclid 头精度（对照，预测显著更高）
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D
from cvfe.episodes import EpisodeSampler

CACHE = r"D:\个人\CVCNN\CVXAI\04_results_from_server\radioml_cache.npz"
CH = (32, 64, 128, 32)
KS = [7, 5, 3, 3]


def conv1d_c(z, wr, wi, pad):
    zp = np.pad(z, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(zp, wr.shape[-1], axis=-1)
    zr, zi = win.real, win.imag
    orr = np.einsum("bclk,ock->bol", zr, wr) - np.einsum("bclk,ock->bol", zi, wi)
    oim = np.einsum("bclk,ock->bol", zr, wi) + np.einsum("bclk,ock->bol", zi, wr)
    return orr + 1j * oim


def modbn(z, gain=None):
    ms = np.mean(np.abs(z) ** 2, axis=(0, 2))
    s = 1.0 / np.sqrt(ms + 1e-4)
    return z * s[None, :, None] if gain is None else z * (s * gain)[None, :, None]


def trunk_forward(z, params, pooling="mean"):
    """z:(B,L) complex -> (B,32)。返回 (emb, last_hidden)。"""
    h = z[:, None, :]
    for i in range(3):
        h = conv1d_c(h, params["wr"][i], params["wi"][i], KS[i] // 2)
        h = modbn(h)
        # modReLU b=0 在初始化处为恒等（实现保持一致：relu(|z|+0)/|z|·z = z）
        B, C, L = h.shape
        h = h.reshape(B, C, L // 2, 2).mean(-1)
    h = conv1d_c(h, params["wr"][3], params["wi"][3], KS[3] // 2)
    h = modbn(h)
    if pooling == "mean":
        emb = h.mean(-1)
    elif pooling == "power":
        emb = np.sqrt(np.mean(np.abs(h) ** 2, axis=-1) + 1e-12).astype(np.complex128)
    return emb, h


def unit(e):
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)


def fisher_ratio(E, y):
    """单位化嵌入的 tr(类间散布)/tr(类内散布)。"""
    mu_all = E.mean(0)
    classes = np.unique(y)
    num, den = 0.0, 0.0
    for c in classes:
        Ec = E[y == c]
        muc = Ec.mean(0)
        num += len(Ec) * np.sum(np.abs(muc - mu_all) ** 2)
        den += np.sum(np.abs(Ec - muc) ** 2)
    return num / max(den, 1e-12)


def protonet_acc(emb_of, sampler, n_ep=200, metric="orbital"):
    accs = []
    Zs, Zq, yq = sampler.sample(n_ep)
    es = unit(emb_of(Zs.reshape(-1, Zs.shape[-1]))).reshape(Zs.shape[:-1] + (-1,))
    eq = unit(emb_of(Zq.reshape(-1, Zq.shape[-1]))).reshape(Zq.shape[:-1] + (-1,))
    mu = es.mean(2)
    if metric == "orbital":
        ip = np.abs(np.einsum("emc,enc->emn", eq.conj(), unit(mu)))
        d2 = -ip
    else:
        d2 = ((eq[:, :, None, :] - mu[:, None, :, :]) ** 2).sum(-1)
    return float((d2.argmin(-1) == yq).mean())


def main():
    t0 = time.time()
    z, y, snr = D.load_radioml(CACHE)
    zn = D.energy_normalize(z).astype(np.complex128)
    rng = np.random.RandomState(19)
    params = {}
    for i, (c_out, c_in, k) in enumerate(zip(CH, [1] + list(CH[:-1]), KS)):
        params["wr"][i] if False else None
    params = {"wr": [rng.randn(co, ci, k) / np.sqrt(ci * k)
                     for co, ci, k in zip(CH, [1] + list(CH[:-1]), KS)],
              "wi": [rng.randn(co, ci, k) / np.sqrt(ci * k)
                     for co, ci, k in zip(CH, [1] + list(CH[:-1]), KS)]}

    # ---- D1: 末层隐层相消比（256 帧，snr>=6）----
    msk = snr >= 6
    idx = np.where(msk)[0]
    sel = rng.choice(idx, 256, replace=False)
    _, hlast = trunk_forward(zn[sel], params)
    coh = np.abs(hlast.mean(-1)).mean()          # |mean_t h| 平均
    mag = np.abs(hlast).mean(-1).mean()          # mean_t|h| 平均
    ratio = float(coh / mag)
    # 逐帧全体相位分布的集中度（R of mean phase across t）
    ph = np.angle(hlast)
    R_t = float(np.abs(np.exp(1j * ph).mean(-1)).mean())

    # ---- Fisher 与 protonet（全部 11 类）----
    cls_all = np.arange(11)
    fidx = []
    for c in cls_all:
        cand = np.where((y == c) & msk)[0]
        fidx.append(rng.choice(cand, 100, replace=False))
    fidx = np.concatenate(fidx)
    fy = y[fidx]
    emb_mean, _ = trunk_forward(zn[fidx], params, "mean")
    emb_pow, _ = trunk_forward(zn[fidx], params, "power")
    fis_mean = fisher_ratio(unit(emb_mean), fy)
    fis_pow = fisher_ratio(unit(emb_pow), fy)

    smp = EpisodeSampler(zn, y, snr, classes=cls_all, n_way=5, k_shot=5,
                         q_per_class=15, seed=19, snr_min=6, snr_max=18)
    acc_mean = protonet_acc(lambda w: trunk_forward(w, params, "mean")[0], smp, 200, "orbital")
    acc_pow = protonet_acc(lambda w: trunk_forward(w, params, "power")[0], smp, 200, "euclid")

    out = {"D1_cancellation_ratio_last": ratio,
           "D1_phase_concentration_R": R_t,
           "chance": 0.2,
           "D2_fisher_meanpool": float(fis_mean),
           "D2_protonet_acc_meanpool_orbital": acc_mean,
           "D3_fisher_powerpool": float(fis_pow),
           "D3_protonet_acc_powerpool_euclid": acc_pow,
           "elapsed_s": round(time.time() - t0, 1)}
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json.dump(out, open(os.path.join(base, "04_results", "logs", "B19_diag_local.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
