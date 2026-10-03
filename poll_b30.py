"""后台轮询：每 5 分钟检查服务器 B30 三臂进程，全部退出后落盘状态并退出。
退出码 0=全部完成；1=异常（有进程消失但 runs 数不足）；2=超时（10h）。
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_server

EXPECTED = 16
T0 = time.time()
while True:
    try:
        c = run_server.connect()
        ci, co, ce = c.exec_command(
            "ps aux | grep -c 'run_b30.py --arms' ; "
            "for d in res_orbit res_hybrid res_gated; do "
            "python3 -c \"import json;d=json.load(open('/tmp/cvfe_work/%s/logs/B30_orbit_pool.json'));print(d['runs'] and len(d['runs']))\" 2>/dev/null || echo 0; done" % "x")
        # 上面占位替换不可靠，改为逐目录读取
        status = {"procs": 0, "counts": {}}
        ci, co, ce = c.exec_command("ps aux | grep 'run_b30.py --arms' | grep -v grep | wc -l")
        status["procs"] = int(co.read().decode().strip() or 0)
        for d in ("res_orbit", "res_hybrid", "res_gated"):
            ci, co, ce = c.exec_command(
                "test -f /tmp/cvfe_work/%s/logs/B30_orbit_pool.json && "
                "%s -c "
                "\"import json;print(len(json.load(open('/tmp/cvfe_work/%s/logs/B30_orbit_pool.json'))['runs']))\" "
                "|| echo 0" % (d, run_server.PYBIN, d))
            status["counts"][d] = int(co.read().decode().strip().split("\n")[0] or 0)
        c.close()
    except Exception as e:
        status = {"procs": -1, "counts": {}, "err": str(e)}
    print(f"[{(time.time()-T0)/60:6.1f}min] {json.dumps(status)}", flush=True)
    done_flag = status["procs"] == 0 and all(
        status.get("counts", {}).get(d, 0) >= EXPECTED for d in
        ("res_orbit", "res_hybrid", "res_gated"))
    bad_flag = status["procs"] == 0 and not done_flag and status.get("counts", {}).get("res_orbit") is not None
    if done_flag:
        print("ALL DONE", flush=True)
        sys.exit(0)
    if bad_flag:
        print("PROCS GONE BUT INCOMPLETE", flush=True)
        sys.exit(1)
    if time.time() - T0 > 10 * 3600:
        print("TIMEOUT", flush=True)
        sys.exit(2)
    time.sleep(300)
