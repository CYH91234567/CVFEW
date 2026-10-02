"""盘点 markdown 中【$...$/$$...$$ 与 `...` 之外】的裸希腊字母/下标/上标记号。"""
import re, sys, collections

GREEK = "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"


def segs_outside_math_code(s):
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
            m = re.match(r"\$[^$\n]*\$", s[i:])
            if m:
                out.append(("math", m.group(0))); i += m.end(); continue
        if s[i] == "`":
            m = re.match(r"`[^`\n]*`", s[i:])
            if m:
                out.append(("code", m.group(0))); i += m.end(); continue
        m = re.match(r"[^$`\n]+", s[i:])
        if not m:
            out.append(("plain", s[i])); i += 1
        else:
            out.append(("plain", m.group(0))); i += m.end()
    return out


for p in sys.argv[1:]:
    s = open(p, encoding="utf8").read()
    cnt = collections.Counter()
    for kind, t in segs_outside_math_code(s):
        if kind != "plain":
            continue
        # 含下划线结构的：σ_θ、mu_c 等（Typora 会被 _ 触发斜体/字面下标）
        for m in re.finditer(r"[A-Za-z" + GREEK + r"]_\{?[A-Za-z0-9" + GREEK + r"]{1,4}\}?", t):
            cnt[m.group(0)] += 1
        # 希腊字母单独出现计数
        for ch in t:
            if ch in GREEK:
                cnt[ch] += 1
        # 组合符帽子（U+0302）与上下标 Unicode
        for m in re.finditer(r"[̂̀́̂]\^?|[₀-₉]+|[⁰-⁹]+", t):
            cnt[m.group(0)] += 1
    print(f"--- {p}")
    for tok, n in cnt.most_common(30):
        print(f"  {n:3d} × {tok}")
