"""B35 n=16 回传：下载服务器两个 midres 扩展目录的 run，按 init 合并去重 →
04_results/logs/B35b_midres.json（与已回传的 B35b_real.json 同枚举口径）。

来源：
  /tmp/cvfe_work/res_b35extmid/logs/B35_midres.json   （原进程：init 47,53,59,61）
  /tmp/cvfe_work/res_b35extmid2/logs/B35_midres.json  （尾部 worker：init 71,79,83,89）
合并后 8 个不重叠 init，与 B35b_real.json 的 8 个 init 逐位配对，供
pool_b35_n16.py 做 16 配对 Wilcoxon。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_server import connect

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS = os.path.join(BASE, "04_results", "logs")
REMOTE_MID = "/tmp/cvfe_work/res_b35extmid/logs/B35_midres.json"
REMOTE_MID2 = "/tmp/cvfe_work/res_b35extmid2/logs/B35_midres.json"


def main():
    c = connect()
    sftp = c.open_sftp()
    tmp = os.path.join(LOGS, "_b35b_mid_raw.json")
    tmp2 = os.path.join(LOGS, "_b35b_mid2_raw.json")
    sftp.get(REMOTE_MID, tmp)
    sftp.get(REMOTE_MID2, tmp2)
    sftp.close()
    c.close()

    merged_runs = {}
    meta = None
    for p in (tmp, tmp2):
        d = json.load(open(p))
        merged_runs.update(d["runs"])
        meta = meta or d.get("meta", {})
    os.remove(tmp)
    os.remove(tmp2)

    # 校验：与 B35b_real 的 init 集逐位相同（8 对 8）
    real = json.load(open(os.path.join(LOGS, "B35b_real.json")))["runs"]
    r_keys = sorted(k.split("_i")[1] for k in real)
    m_keys = sorted(k.split("_i")[1] for k in merged_runs)
    assert len(merged_runs) == 8, f"midres merged n={len(merged_runs)} != 8"
    assert m_keys == r_keys, f"init mismatch: mid {m_keys} vs real {r_keys}"

    out = {"meta": {"prereg": "PREREG_B35.md n=16 extension (merged two midres workers)",
                    "source": "res_b35extmid (47,53,59,61) + res_b35extmid2 (71,79,83,89)",
                    **{k: v for k, v in meta.items() if k != "prereg"}},
           "runs": merged_runs}
    p = os.path.join(LOGS, "B35b_midres.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"B35b_midres.json: {len(merged_runs)} runs -> {p}")
    print("inits:", m_keys)


if __name__ == "__main__":
    main()
