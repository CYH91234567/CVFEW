"""检查服务器端 B35 n=16 扩展（INIT_SEEDS[8:16]）状态，不做任何修改。"""
import sys, paramiko
sys.path.insert(0, ".")
from run_server import connect

c = connect()


def sh(cmd):
    _, o, e = c.exec_command(cmd, timeout=60)
    print(f"$ {cmd}")
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace").strip()
    if err:
        print("STDERR:", err)
    print("-" * 60)


sh("ls -la --time-style=full-iso /tmp/cvfe_work/res/logs/ 2>&1 | grep -i b35")
sh("ps -eo pid,etime,stat,cmd | grep -i -E 'b35|nohup' | grep -v grep")
sh("ls -la --time-style=full-iso /tmp/cvfe_work/code/ 2>&1 | grep -i -E 'b35|pool'")
c.close()
