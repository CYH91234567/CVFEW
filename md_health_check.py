"""扫描项目 markdown 的 Typora 渲染健康度（转义感知）：
1. 控制字符（TAB/CR/BEL 等 C0）
2. 表格列数不一致（按未转义的 | 分割）
3. 行内 $ 数学定界符配平（排除代码块与 $$ 显示块）
"""
import os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CTRL = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]")
PIPE_SPLIT = re.compile(r"(?<!\\)\|")
MATH = re.compile(r"(?<!\\)\$")


def cells(line):
    return PIPE_SPLIT.split(line.strip())


def scan(p):
    s = open(p, encoding="utf8").read()
    hits = []
    for m in CTRL.finditer(s):
        hits.append(("CTRL", m.start(), repr(m.group(0))))
    s2 = re.sub(r"```.*?```", "", s, flags=re.S)
    s2 = re.sub(r"\$\$.*?\$\$", "", s2, flags=re.S)
    s2 = re.sub(r"`[^`]*`", "", s2)                  # 内联代码 span（Typora 按字面渲染）
    if len(MATH.findall(s2)) % 2:
        hits.append(("UNBALANCED_$", 0, f"count={len(MATH.findall(s2))}"))
    lines = s.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            hdr = cells(lines[i])
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = cells(lines[j])
                if len(row) != len(hdr):
                    hits.append(("TABLE_COLS", j + 1,
                                 f"hdr={len(hdr)} row={len(row)}: {lines[j][:60]}"))
                j += 1
            i = j
        else:
            i += 1
    return hits


bad = 0
for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    for f in sorted(files):
        if not f.endswith(".md"):
            continue
        p = os.path.join(root, f).replace("\\", "/")
        hits = scan(p)
        if hits:
            bad += 1
            print(f"### {p}  ({len(hits)})")
            for kind, pos, ctx in hits[:8]:
                print(f"   [{kind}] {ctx}")
print(f"\nFiles with REAL issues: {bad}")
