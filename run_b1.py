"""B1 入口：本地诊断包（numpy only）。

用法： python run_b1.py --cache <radioml_cache.npz> --out <04_results路径>
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import diag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
    t0 = time.time()
    diag.run_b1(a.cache, out, seed=a.seed)
    print(f"B1 DONE in {time.time()-t0:.0f}s")
