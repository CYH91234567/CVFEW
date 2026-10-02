"""从公开镜像（dannis999/RML2016.10b）重建 RadioML 2016.10b cache（第二真实数据集）。

与 make_cache.py（10a）同构：仅取项目用到的 SNR {0,6,12,18} × 10 类 = 40 文件。
10b 与 10a 的差异：无 AM-SSB（10 类），样本生成更"真实/困难"（DeepSig 官方设计）。
产物：/tmp/cvfe_work/code/radioml_cache_b10b.npz，含 z (N,128) complex64、
y (N,) int64（类索引按 CLASSES_B 顺序）、snr (N,) int64。
解析逻辑与 CVXAI/cvx/radioml.py:_parse_one 逐字一致（镜像重建，非原始副本）。
"""
import os, re, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 10b 的类顺序：8 个数字类 + 2 个模拟类（无 AM-SSB）
CLASSES_B = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4",
             "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = [0, 6, 12, 18]
BASE = "https://raw.githubusercontent.com/dannis999/RML2016.10b/master/2016.10b"
RAW = "/tmp/cvfe_work/raw_b10b"
CACHE = "/tmp/cvfe_work/code/radioml_cache_b10b.npz"
TOK = re.compile(r"[^()\s]+")


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
        return fn, os.path.getsize(p), "cached"
    url = BASE + "/" + urllib.parse.quote(fn)
    err = ""
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, p)
            if os.path.getsize(p) > 3_000_000:
                return fn, os.path.getsize(p), "ok"
        except Exception as e:
            err = str(e)[:60]
    return fn, os.path.getsize(p) if os.path.exists(p) else 0, f"FAIL {err}"


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    files = [f"{cl} {snr}.txt" for cl in CLASSES_B for snr in SNRS]
    print(f"下载 {len(files)} 个文件（并行 6）...", flush=True)
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fn, sz, st in ex.map(fetch, files):
            print(f"  {fn}: {sz // 1000} KB [{st}]", flush=True)
            if st not in ("ok", "cached"):
                fails.append(fn)
    if fails:
        print("缺失文件:", fails)
        sys.exit(1)
    zs, ys, ss = [], [], []
    for ci, cl in enumerate(CLASSES_B):
        for snr in SNRS:
            arr = parse_one(os.path.join(RAW, f"{cl} {snr}.txt"))
            if arr.shape[0] != 1000 or arr.shape[1] != 128:
                print(f"[warn] {cl} {snr} shape={arr.shape}")
            zs.append(arr)
            ys.append(np.full(arr.shape[0], ci, dtype=np.int64))
            ss.append(np.full(arr.shape[0], snr, dtype=np.int64))
    z = np.concatenate(zs); y = np.concatenate(ys); snr = np.concatenate(ss)
    np.savez(CACHE, z=z, y=y, snr=snr)
    import collections
    print(f"cache saved: z{z.shape} {z.dtype}", flush=True)
    print("per-class:", dict(sorted(collections.Counter(y.tolist()).items())))
    print("snr set:", sorted(set(snr.tolist())))
    print("|z| mean %.4f, 非有限点 %d" % (np.abs(z).mean(), int((~np.isfinite(z)).sum())))
    print("CACHE_B10B_OK", CACHE)


if __name__ == "__main__":
    main()
