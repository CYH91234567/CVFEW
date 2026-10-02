"""本地备选 B34-P1：第二数据集（RML2016.10b）上 Thm1 实数据验证（确定性方法）。

服务器不可达时的备选路径：本地下载 10b 数据子集（raw.githubusercontent 可达），
本地 numpy 跑确定性方法（PhaseMAP-ML/轨道/欧氏）注入网格，验证 P1：
  轨道/PhaseMAP 注入平坦 vs 欧氏下降（10a 对照：0.0 vs −12.2）。
SOTA（P2/P3）需 torch 训练，仍待服务器恢复后由 run_b34.py 完成。
评估量缩减为 12 格 × 300 episodes（P1 判定足够；确定性方法无训练方差）。
"""
import json
import os, re, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler

CLASSES_B = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4",
             "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = [0, 6, 12, 18]
BASE = "https://raw.githubusercontent.com/dannis999/RML2016.10b/master/2016.10b"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "04_results", "local_b10b")
RAW = os.path.join(LOCAL_DIR, "raw")
CACHE = os.path.join(LOCAL_DIR, "cache_b10b.npz")
TOK = re.compile(r"[^()\s]+")
N_EVAL = 300


def parse_one(path):
    with open(path, "r") as f:
        txt = f.read()
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        toks = TOK.findall(line)
        if len(toks) < 8:
            continue
        try:
            rows.append([complex(t) for t in toks])
        except ValueError:
            continue
    return np.array(rows, dtype=np.complex64)


def fetch(fn):
    p = os.path.join(RAW, fn)
    if os.path.exists(p) and os.path.getsize(p) > 3_000_000:
        return fn, "cached"
    url = BASE + "/" + urllib.parse.quote(fn)
    for _ in range(2):
        try:
            urllib.request.urlretrieve(url, p)
            if os.path.getsize(p) > 3_000_000:
                return fn, "ok"
        except Exception:
            pass
    return fn, "FAIL"


def build_cache():
    os.makedirs(RAW, exist_ok=True)
    files = [f"{cl} {snr}.txt" for cl in CLASSES_B for snr in SNRS]
    print(f"下载 {len(files)} 文件（并行 8，本地）...", flush=True)
    fails = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fn, st in ex.map(fetch, files):
            if st == "FAIL":
                fails.append(fn)
                print(f"  {fn}: FAIL", flush=True)
    if fails:
        print("缺失:", fails)
        return False
    zs, ys, ss = [], [], []
    for ci, cl in enumerate(CLASSES_B):
        for snr in SNRS:
            arr = parse_one(os.path.join(RAW, f"{cl} {snr}.txt"))
            zs.append(arr)
            ys.append(np.full(arr.shape[0], ci, dtype=np.int64))
            ss.append(np.full(arr.shape[0], snr, dtype=np.int64))
    z = np.concatenate(zs); y = np.concatenate(ys); snr = np.concatenate(ss)
    np.savez(CACHE, z=z, y=y, snr=snr)
    print(f"cache built: z{z.shape}", flush=True)
    return True


def main():
    if not os.path.exists(CACHE):
        if not build_cache():
            sys.exit(1)
    z, y, snr = D.load_radioml(CACHE)
    zn = D.energy_normalize(z).astype(np.complex64)
    print(f"load {z.shape}", flush=True)
    ALL = list(range(10))
    inj_grid = [("none", 0.0), ("global", np.pi / 6), ("global", np.pi / 2)]
    results = []
    for sl in SNRS:
        for mode, strength in inj_grid:
            smp = EpisodeSampler(zn, y, snr, classes=ALL, n_way=5, k_shot=5,
                                 q_per_class=15, seed=7000 + 7 * sl,
                                 snr_min=sl, snr_max=sl)
            inj = None if mode == "none" else (mode, strength)
            Zs, Zq, yq = smp.sample(N_EVAL, inject=inj)
            Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
            acc = {}
            acc["euclid"] = float((E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean(1).mean())
            acc["orbital"] = float((E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean(1).mean())
            acc_pm = np.zeros(N_EVAL)
            for g0 in range(0, N_EVAL, 100):
                s_ = slice(g0, min(g0 + 100, N_EVAL))
                mu, aux = E.phasemap_em(Zs128[s_])
                acc_pm[s_] = (E.cls_marginal(Zq128[s_], mu, aux) == yq[s_]).mean(1)
            acc["phasemap_ml"] = float(acc_pm.mean())
            results.append({"dataset": "RML2016.10b-local", "split": "all10", "snr": sl,
                            "inject_mode": mode, "inject_strength": float(strength),
                            "acc": acc, "n_eval": N_EVAL})
            print(f"  [SNR{sl} {mode}{strength:.2f}] " +
                  " ".join(f"{k}={100*v:.1f}" for k, v in acc.items()), flush=True)
    out = os.path.join(LOCAL_DIR, "B34_det_local.json")
    json.dump(results, open(out, "w"), indent=1)
    print("LOCAL_B34_DET_DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
