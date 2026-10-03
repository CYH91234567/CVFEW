"""服务器执行辅助（CVFEW 版）：上传代码+数据 -> 远程执行 -> 回传结果 -> 清理。

用法:
  python run_server.py upload                 # 上传代码与数据缓存
  python run_server.py run "<命令>"           # 远程执行（流式输出）
  python run_server.py fetch                  # 回传结果到本机 04_results
  python run_server.py clean                  # 删除服务器上全部项目文件

【凭据纪律】连接参数一律从环境变量 CVFE_HOST/CVFE_PORT/CVFE_USER/CVFE_PWD
读取；本地的 servercreds.py（git 忽略）作为开发回退。凭据不得进入版本库
（历史教训：明文凭据曾随首个 commit 推送到公开仓并泄露，见 HANDOFF 警示条目）。
"""
import os, sys, stat
import paramiko

LOCAL_CODE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/tmp/cvfe_work/code"
LOCAL_RES = os.path.abspath(os.path.join(LOCAL_CODE, "..", "04_results"))
PYBIN = "/home/kjds512/anaconda3/envs/msb/bin/python"


def _cred(name, env):
    v = os.environ.get(env)
    if v:
        return v
    try:
        from servercreds import CREDS  # 本地未跟踪回退（.gitignore）
        return CREDS[name]
    except ImportError:
        raise SystemExit(
            f"服务器凭据缺失：请设环境变量 {env} 或提供本地 servercreds.py（见模块文档）")


HOST = _cred("host", "CVFE_HOST")
PORT = int(_cred("port", "CVFE_PORT"))
USER = _cred("user", "CVFE_USER")
PWD = _cred("pwd", "CVFE_PWD")


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PWD, timeout=20,
              allow_agent=False, look_for_keys=False)
    return c


def upload(c):
    c.exec_command(f"mkdir -p {REMOTE}/cvfe /tmp/cvfe_work/res/logs "
                   f"/tmp/cvfe_work/res/tables /tmp/cvfe_work/res/figures")[1].read()
    sftp = c.open_sftp()
    files = []
    for f in os.listdir(LOCAL_CODE):
        if f.endswith(".py"):
            files.append((os.path.join(LOCAL_CODE, f), f"{REMOTE}/{f}"))
    for f in os.listdir(os.path.join(LOCAL_CODE, "cvfe")):
        if f.endswith(".py"):
            files.append((os.path.join(LOCAL_CODE, "cvfe", f), f"{REMOTE}/cvfe/{f}"))
    cache = os.environ.get("CVFE_CACHE",
                           os.path.join(LOCAL_CODE, "..", "..", "CVXAI",
                                        "04_results_from_server", "radioml_cache.npz"))
    files.append((cache, f"{REMOTE}/radioml_cache.npz"))
    for lp, rp in files:
        sftp.put(lp, rp)
        print("put", os.path.basename(lp))
    sftp.close()


def run(c, cmd, timeout=3600, stream=True):
    chan = c.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    while True:
        if chan.recv_ready():
            b = chan.recv(65536)
            if not b:
                break
            out += b
            if stream:
                sys.stdout.write(b.decode("utf-8", "replace"))
                sys.stdout.flush()
        elif chan.recv_stderr_ready():
            b = chan.recv_stderr(65536)
            if not b:
                break
            out += b
            if stream:
                sys.stderr.write(b.decode("utf-8", "replace"))
                sys.stderr.flush()
        elif chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(65536)
            break
        else:
            import time
            time.sleep(0.5)
    return chan.recv_exit_status(), out.decode("utf-8", "replace")


def fetch(c):
    sftp = c.open_sftp()
    got = []

    def walk(d):
        for e in sftp.listdir_attr(d):
            p = f"{d}/{e.filename}"
            if stat.S_ISDIR(e.st_mode):
                walk(p)
            else:
                rel = os.path.relpath(p, "/tmp/cvfe_work/res")
                lp = os.path.join(LOCAL_RES, rel)
                os.makedirs(os.path.dirname(lp), exist_ok=True)
                sftp.get(p, lp)
                got.append(lp)
    walk("/tmp/cvfe_work/res")
    sftp.close()
    print("fetched:", len(got))


def clean(c):
    _, out, _ = c.exec_command("rm -rf /tmp/cvfe_work && echo cleaned")
    print(out.read().decode())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    c = connect()
    if cmd == "upload":
        upload(c)
    elif cmd == "run":
        st, _ = run(c, f"cd {REMOTE} && {PYBIN} " + " ".join(sys.argv[2:]))
        print(f"\n[exit={st}]")
    elif cmd == "fetch":
        fetch(c)
    elif cmd == "clean":
        clean(c)
    c.close()
