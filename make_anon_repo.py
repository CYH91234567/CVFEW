"""B-J7：构建匿名代码仓快照（双盲交付）。

规则（gap-tsp-tnnls B-J7 / TNNLS double-anonymous）：
- 复制科学代码（cvfe/ + run_*/analyze_*/make_table*）与预注册文档；
- 排除基础设施与身份相关文件：run_server.py、servercreds*、poll_*、
  check_b35ext、launch_*、__pycache__、*.log、smoketest/、.git；
- 全树身份扫描（用户名/机器路径/服务器地址/handle/仓库名），命中即报错中止；
- 输出为独立目录 + 单一匿名提交（作者 Anonymous），不带 remote。
  用户选定匿名托管（anonymous.4open.science 或匿名 GitHub 账号）后，
  在输出目录执行 git remote add + push 即可。

用法：python make_anon_repo.py [--out ../anon_release]
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ANON_NAME = "phase-quotient-few-shot"   # 中性仓库名

# 排除（基础设施 / 身份 / 环境噪声）
EXCLUDE_FILES = {
    "run_server.py", "servercreds.py", "servercreds.py.template",
    "make_anon_repo.py", "launch_round19.sh", "check_b35ext.py",
}
EXCLUDE_PREFIXES = ("poll_",)
EXCLUDE_SUFFIXES = (".log", ".pyc")

# 身份模式（命中即失败）
IDENTITY_PATTERNS = [
    "个人/CVCNN", "C:/个人", "D:/个人", "C:\\Users\\27067", "D:\\个人",
    "kjds512", "10.12.149", "100029", "CYH91234567", "github.com/CYH",
    "27067@",
]


def should_include(fn):
    if fn in EXCLUDE_FILES:
        return False
    if any(fn.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    if any(fn.endswith(s) for s in EXCLUDE_SUFFIXES):
        return False
    return True


def scan_identity(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    s = f.read()
            except OSError:
                continue
            for pat in IDENTITY_PATTERNS:
                if pat in s:
                    hits.append((os.path.relpath(p, root), pat))
    return hits


def main():
    out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1]
                          if "--out" in sys.argv
                          else os.path.join(BASE, "..", "anon_release"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "code"))
    os.makedirs(os.path.join(out, "prereg"))

    # 1) 代码
    n_py = 0
    os.makedirs(os.path.join(out, "code", "cvfe"), exist_ok=True)
    for fn in sorted(os.listdir(BASE)):
        if not should_include(fn):
            continue
        p = os.path.join(BASE, fn)
        if os.path.isfile(p) and (fn.endswith(".py") or fn in (".gitignore",)):
            shutil.copy2(p, os.path.join(out, "code", fn))
            n_py += 1
    pkg = os.path.join(BASE, "cvfe")
    for fn in sorted(os.listdir(pkg)):
        if fn.endswith(".py"):
            shutil.copy2(os.path.join(pkg, fn), os.path.join(out, "code", "cvfe", fn))
            n_py += 1
    # 2) 预注册（可复现性佐证；先按行清洗身份/服务器状态引用）
    n_pr = 0
    dropped = 0
    plan = os.path.join(BASE, "..", "02_plan")
    for fn in sorted(os.listdir(plan)):
        if not (fn.startswith("PREREG_") and fn.endswith(".md")):
            continue
        with open(os.path.join(plan, fn), encoding="utf-8") as f:
            lines = f.readlines()
        keep, n_drop = [], 0
        for ln in lines:
            if any(pat in ln for pat in IDENTITY_PATTERNS):
                n_drop += 1
                continue
            keep.append(ln)
        with open(os.path.join(out, "prereg", fn), "w", encoding="utf-8") as f:
            f.writelines(keep)
        n_pr += 1
        dropped += n_drop
    print(f"prereg sanitize: dropped {dropped} identity/status lines")

    # 3) README（匿名版，脚本内模板，覆盖式写入）
    readme = os.path.join(BASE, "ANON_README.md")
    if os.path.exists(readme):
        shutil.copy2(readme, os.path.join(out, "README.md"))

    # 4) 身份扫描（命中即失败中止）
    hits = scan_identity(out)
    if hits:
        shutil.rmtree(out)
        raise SystemExit(f"IDENTITY HITS — aborting:\n" +
                         "\n".join(f"  {p}: {pat}" for p, pat in hits[:20]))

    print(f"copied {n_py} py files, {n_pr} preregs -> {out}")

    # 4) 单一匿名提交（无 remote）
    r = subprocess.run(["git", "init", "-q"], cwd=out)
    subprocess.run(["git", "add", "-A"], cwd=out, check=True)
    subprocess.run(["git", "-c", "user.name=Anonymous",
                    "-c", "user.email=anonymous@invalid",
                    "commit", "-qm",
                    "Anonymous submission code: few-shot learning on complex "
                    "representations modulo global phase; equivariant trunk + "
                    "phase-invariant readout + certificate suite"], cwd=out, check=True)
    print("anonymous single commit created (no remote). Next:")
    print(f"  cd {out}")
    print("  git remote add origin <anonymous-hosting-url>")
    print("  git push -u origin main   # 或按托管平台指引")
    print(f"identity scan: clean; tree: {out}")


if __name__ == "__main__":
    main()
