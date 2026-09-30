import json, glob, os, subprocess
DIRS = {"orbit": "/tmp/cvfe_work/res_orbit/logs/B30_orbit_pool.json",
        "hybrid": "/tmp/cvfe_work/res_hybrid/logs/B30_orbit_pool.json",
        "gated": "/tmp/cvfe_work/res_gated/logs/B30_orbit_pool.json",
        "orbitL0.3": "/tmp/cvfe_work/res_orb03/logs/B30_orbit_pool.json",
        "learnlam": "/tmp/cvfe_work/res_learnlam/logs/B30_orbit_pool.json"}
for name, p in DIRS.items():
    if os.path.exists(p):
        d = json.load(open(p))
        runs = d["runs"]
        vals = []
        for k, v in runs.items():
            e = v["eval"].get("valsel")
            if e:
                vals.append(round(100 * sum(e[f"sigma_{s:.2f}"]["orbital"] for s in
                                            (0.0, 1.0471975511965976, 3.141592653589793)) / 3, 1))
        print(f"{name:10s} n={len(runs):2d} valsel={vals}")
    else:
        print(f"{name:10s} n= 0")
n = subprocess.run("ps aux | grep 'run_b30.py --arms' | grep -v grep | wc -l",
                   shell=True, capture_output=True, text=True).stdout.strip()
print("procs:", n)
