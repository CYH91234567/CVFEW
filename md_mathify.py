"""把 markdown 中【$...$ 数学 span 与 `...` 代码 span 之外】的裸数学记号
包成 $...$（Typora/KaTeX 正常渲染，且避免 _ 触发斜体解析）。

仅转换：希腊字母前缀的下标/帽子、以及常见完整表达式。
不转换：拉丁前缀的运行标签（m_hard / l_v2 / big_es3200 等）、文件名、希腊字母单字。
"""
import re, sys

GREEK = "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"

# 顺序敏感：长的先替换
RULES = [
    (r"σ_θ̂", r"$\\hat\\sigma_\\theta$"),
    (r"σ̂²", r"$\\hat\\sigma^2$"),
    (r"σ̂", r"$\\hat\\sigma$"),
    (r"γ̂", r"$\\hat\\gamma$"),
    (r"κ̂", r"$\\hat\\kappa$"),
    (r"μ̂", r"$\\hat\\mu$"),
    (r"δ_inv", r"$\\delta_{\\mathrm{inv}}$"),
    (r"σ_θ=π/6", r"$\\sigma_\\theta = \\pi/6$"),
    (r"σ_θ=π/3", r"$\\sigma_\\theta = \\pi/3$"),
    (r"σ_θ=π/2", r"$\\sigma_\\theta = \\pi/2$"),
    (r"σ_θ=π", r"$\\sigma_\\theta = \\pi$"),
    (r"σ_θ→0", r"$\\sigma_\\theta \\to 0$"),
    (r"σ_θ→π", r"$\\sigma_\\theta \\to \\pi$"),
    (r"σ_θ∈\[0,π\]", r"$\\sigma_\\theta \\in [0,\\pi]$"),
    (r"σ_θ", r"$\\sigma_\\theta$"),
    (r"θ_j", r"$\\theta_j$"),
    (r"g₁", r"$g_1$"),
    (r"g₂", r"$g_2$"),
]
RULES = [(re.compile(p), r) for p, r in RULES]


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


def convert(s):
    res, changed = [], 0
    for kind, t in segs(s):
        if kind != "plain":
            res.append(t)
            continue
        for pat, repl in RULES:
            t, n = pat.subn(repl, t)
            changed += n
        res.append(t)
    return "".join(res), changed


total = 0
for p in sys.argv[1:]:
    s = open(p, encoding="utf8").read()
    s2, n = convert(s)
    if n:
        open(p, "w", encoding="utf8").write(s2)
    total += n
    print(f"{p}: {n} tokens wrapped")
print("total:", total)
