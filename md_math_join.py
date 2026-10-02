"""把跨行的内联数学段（$...$ 内含换行）合并为单行（Typora 对跨行内联数学支持不稳）。
用与 md_mathify 相同的 tokenizer 精确定位数学段，避免把相邻两个独立数学段误判。"""
import re, sys


def segs(s):
    out, i = [], 0
    while i < len(s):
        m = re.match(r"```.*?```", s[i:], re.S)
        if m:
            out.append(("code", m.group(0))); i += m.end(); continue
        if s.startswith("$$", i):
            m = re.match(r"\$\$.*?\$\$", s[i:], re.S)
            if m:
                out.append(("math", m.group(0))); i += m.end(); continue
        if s[i] == "$":
            m = re.match(r"\$[^$]*\$", s[i:], re.S)
            if m:
                out.append(("math", m.group(0))); i += m.end(); continue
        if s[i] == "`":
            m = re.match(r"`[^`]*`", s[i:], re.S)
            if m:
                out.append(("code", m.group(0))); i += m.end(); continue
        m = re.match(r"[^$`]+", s[i:])
        if not m:
            out.append(("plain", s[i])); i += 1
        else:
            out.append(("plain", m.group(0))); i += m.end()
    return out


total = 0
for p in sys.argv[1:]:
    s = open(p, encoding="utf8").read()
    res = []
    n = 0
    for kind, t in segs(s):
        if kind == "math" and t.startswith("$") and not t.startswith("$$") and "\n" in t:
            t = re.sub(r"\s*\n\s*", " ", t).strip()
            n += 1
        res.append(t)
    if n:
        open(p, "w", encoding="utf8").write("".join(res))
    total += n
    print(f"{p}: {n} multiline inline-math spans joined")
print("total:", total)
