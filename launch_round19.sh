#!/bin/bash
# 第十九轮上机启动脚本：B40 四臂（10b 学习层，B-J1）+ B35 midres 尾部并行 worker。
# 每个 worker 独立日志 + 独立 json（B40_{arm}.json 按臂分文件，无写冲突）。
cd /tmp/cvfe_work/code

launch() {  # name, cmd...
  local name="$1"; shift
  if pgrep -f "$name" > /dev/null; then
    echo "RUNNING: $name"
  else
    OMP_NUM_THREADS=1 nohup "$@" > "/tmp/cvfe_work/${name}.log" 2>&1 < /dev/null &
    echo "LAUNCHED: $name (pid $!)"
  fi
}

launch b40_gated  python3 run_b40.py --out /tmp/cvfe_work/res_b40 --arms gated  --n-inits 8
launch b40_comp   python3 run_b40.py --out /tmp/cvfe_work/res_b40 --arms comp   --n-inits 8
launch b40_midres python3 run_b40.py --out /tmp/cvfe_work/res_b40 --arms midres --n-inits 8
launch b35_midres2 python3 run_b35.py --out /tmp/cvfe_work/res_b35extmid2 --arms midres --init-offset 12 --n-inits 4 --mode full

sleep 2
ps -eo pid,etime,cmd | grep -E "run_b40|run_b35" | grep -v grep
