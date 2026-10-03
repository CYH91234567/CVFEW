"""后台轮询 v2：每 5 分钟检查服务器 B30 五个臂（orbit/hybrid/gated/orbitL0.3/
learnlam，各独立 --out 目录），全部进程退出后落盘状态并退出。
退出码 0=全部完成（每臂 16 runs）；1=进程消失但 runs 不足；2=超时（14h）。
"""
import json, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import run_server

ARMS = [("res_orbit", "orbit"), ("res_hybrid", "hybrid"), ("res_gated", "gated"),
        ("res_orb03", "orbitL0.3"), ("res_learnlam", "learnlam")]
EXPECTED = 16
T0 = time.time()
while True:
    status = {"procs": 0, "counts": {}}
    try:
        c = run_server.connect()
        ci, co, ce = c.exec_command("ps aux | grep 'run_b30.py --arms' | grep -v grep | wc -l")
        status["procs"] = int(co.read().decode().strip() or 0)
        for d, name in ARMS:
            ci, co, ce = c.exec_command(
                "test -f /tmp/cvfe_work/%s/logs/B30_orbit_pool.json && "
                "%s -c "
                "\"import json;d=json.load(open('/tmp/cvfe_work/%s/logs/B30_orbit_pool.json'));"
                "print(len(d['runs']))\" || echo 0" % (d, run_server.PYBIN, d))
            status["counts"][name] = int(co.read().decode().strip().split("\n")[0] or 0)
        c.close()
    except Exception as e:
        status = {"procs": -1, "counts": {}, "err": str(e)}
    print(f"[{(time.time()-T0)/60:6.1f}min] {json.dumps(status)}", flush=True)
    counts = status.get("counts", {})
    all_done = (status["procs"] == 0 and len(counts) == len(ARMS)
                and all(counts.get(n, 0) >= EXPECTED for _, n in ARMS))
    procs_gone = status["procs"] == 0 and status.get("counts")
    if all_done:
        print("ALL_DONE", flush=True)
        sys.exit(0)
    if procs_gone and not all_done:
        print("PROCS_GONE_INCOMPLETE " + json.dumps(counts), flush=True)
        sys.exit(1)
    if time.time() - T0 > 14 * 3600:
        print("TIMEOUT", flush=True)
        sys.exit(2)
    time.sleep(300)
