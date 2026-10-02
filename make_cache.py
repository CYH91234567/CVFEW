"""从公开镜像（dannis999/RML2016.10a，DeepSig 标准文本格式）重建 RadioML cache。

产物：/tmp/cvfe_work/code/radioml_cache.npz，含 z (N,128) complex64、
y (N,) int64（类索引，顺序同 cvfe.episodes.MOD_CLASSES）、snr (N,) int64。
仅取项目用到的 SNR {0,6,12,18} × 11 类 = 44 文件（各 1000 帧）。
解析逻辑与 CVXAI/cvx/radioml.py:_parse_one 逐字一致。
注意：原 cache 已随服务器清零遗失；本 cache 由公开镜像重建，
逐位一致性无法对原副本验证，结果按"镜像重建数据"如实标注。
"""
import os, re, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CLASSES = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
SNRS = [0, 6, 12, 18]
BASE = "https://raw.githubusercontent.com/dannis999/RML2016.10a/master/2016.10a"
RAW = "/tmp/cvfe_work/raw"
CACHE = "/tmp/cvfe_work/code/radioml_cache.npz"
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
    for attempt in range(3):
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
    files = [f"{cl} {snr}.txt" for cl in CLASSES for snr in SNRS]
    print(f"下载 {len(files)} 个文件（并行 6）...", flush=True)
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fn, sz, st in ex.map(fetch, files):
            print(f"  {fn}: {sz // 1000} KB [{st}]", flush=True)
            if st != "ok" and st != "cached":
                fails.append(fn)
    if fails:
        print("缺失文件:", fails)
        sys.exit(1)

    zs, ys, ss = [], [], []
    for ci, cl in enumerate(CLASSES):
        for snr in SNRS:
            arr = parse_one(os.path.join(RAW, f"{cl} {snr}.txt"))
            if arr.shape[0] != 1000 or arr.shape[1] != 128:
                print(f"[warn] {cl} {snr} shape={arr.shape} （非 (1000,128)）")
            zs.append(arr)
            ys.append(np.full(arr.shape[0], ci, dtype=np.int64))
            ss.append(np.full(arr.shape[0], snr, dtype=np.int64))
    z = np.concatenate(zs); y = np.concatenate(ys); snr = np.concatenate(ss)
    np.savez(CACHE, z=z, y=y, snr=snr)
    print(f"cache saved: z{z.shape} {z.dtype} y{y.shape} snr{snr.shape}", flush=True)
    # 自检
    assert z.shape[0] == y.shape[0] == snr.shape[0]
    import collections
    cnt = collections.Counter(y.tolist())
    print("per-class:", dict(sorted(cnt.items())))
    print("snr set:", sorted(set(snr.tolist())))
    print("|z| mean %.4f, 非有限点数 %d" % (np.abs(z).mean(), int((~np.isfinite(z)).sum())))
    print("CACHE_OK", CACHE)


if __name__ == "__main__":
    main()
