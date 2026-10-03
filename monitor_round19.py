"""B35 n=16 + B40 四臂进度监控（后台轮询，每 15 分钟一次，写本地日志）。

输出：03_code/monitor_round19.log（valsel 完成数 + 进程数 + 关键事件）。
事件：B35 midres 原进程完成 i61（应 kill）；各臂全部完成。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_server import connect

LOGS = [
    ("b40_real",      8, "/tmp/cvfe_work/res_b40/logs/B40_real.json"),
    ("b40_gated",     8, "/tmp/cvfe_work/res_b40/logs/B40_gated.json"),
    ("b40_comp",      8, "/tmp/cvfe_work/res_b40/logs/B40_comp.json"),
    ("b40_midres",    8, "/tmp/cvfe_work/res_b40/logs/B40_midres.json"),
    ("b35_midres2",   4, "/tmp/cvfe_work/res_b35extmid2/logs/B35_midres.json"),
    ("b35_midres",    8, "/tmp/cvfe_work/res_b35extmid/logs/B35_midres.json"),
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_round19.log")
B35_ORIG_PID = "1078895"  # 原始 midres 进程，i61 完成后应杀掉（避免与尾部 worker 重复 i71+）


def status():
    try:
        c = connect()
        _, o, _ = c.exec_command(
            "date '+%H:%M:%S'; cat /proc/loadavg; ps -eo pid,cmd | grep -cE 'run_b4[05].py'",
            timeout=30)
        head = o.read().decode("utf-8", "replace").strip().split("\n")
        rows = []
        for name, want, path in LOGS:
            _, o2, _ = c.exec_command(
                f"test -f {path} && python3 -c \"import json;print(len(json.load(open('{path}'))['runs']))\" || echo 0",
                timeout=30)
            n = o2.read().decode("utf-8", "replace").strip().split("\n")[0]
            try:
                n = int(n)
            except ValueError:
                n = -1
            rows.append(f"{name}={n}/{want}")
            if n == want:
                rows.append(f"** {name} COMPLETE **")
        # B35 原进程是否仍在 i61（valsel=4 时 = i61 完成，应 kill）
        _, o3, _ = c.exec_command(
            f"ps -p {B35_ORIG_PID} -o etime --no-headers 2>/dev/null || echo DEAD", timeout=20)
        et = o3.read().decode().strip()
        alive = "alive" if et != "DEAD" else "dead"
        rows.append(f"b35_orig({alive}: {et})")
        c.close()
        return head + [" | ".join(rows)]
    except Exception as e:
        return [f"ERR {type(e).__name__}: {e}"]


def control(counts, lines):
    """流水线控制（幂等，重复执行安全）：
    1) B35 原始 midres 进程在 i61（第 4 个）完成后 kill（尾部 worker 接管 i71+）；
    2) b40_real 完成后 SIGCONT 唤醒 gated；gated 完成后唤醒 comp
       （并发降为 ~3，避免 GPU 命令队列超线性退化）。"""
    actions = []
    try:
        c = connect()
    except Exception as e:
        return [f"control connect ERR: {e}"]
    try:
        if counts.get("b35_midres", 0) >= 4:
            _, o, _ = c.exec_command(f"kill {B35_ORIG_PID} 2>/dev/null && echo KILLED_B35_ORIG "
                                      f"|| echo b35_orig_already_gone", timeout=20)
            actions.append(o.read().decode().strip())
        if counts.get("b40_real", 0) >= 8:
            _, o, _ = c.exec_command("kill -CONT 1109203 2>/dev/null && echo CONT_GATED "
                                     "|| echo gated_not_stopped", timeout=20)
            actions.append(o.read().decode().strip())
        if counts.get("b40_gated", 0) >= 8:
            _, o, _ = c.exec_command("kill -CONT 1109581 2>/dev/null && echo CONT_COMP "
                                     "|| echo comp_not_stopped", timeout=20)
            actions.append(o.read().decode().strip())
    finally:
        c.close()
    return actions


def main():
    notified = set()
    while True:
        lines = status()
        counts = {row.split("=")[0].strip(): int(row.split("=")[1].split("/")[0])
                  for row in lines[-1].split(" | ") if "=" in row and "orig" not in row}
        actions = control(counts, lines)
        msg = time.strftime("%Y-%m-%d %H:%M") + " :: " + " ; ".join(lines)
        if actions:
            msg += " ; ACTIONS: " + ", ".join(a for a in actions if a)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)
        for ln in lines:
            for name, want, _ in LOGS:
                if f"= {want}/{want}" in ln:
                    notified.add(name)
        if len(notified) >= 5:  # b35_midres（原进程）不在等
            break
        time.sleep(900)


if __name__ == "__main__":
    main()
