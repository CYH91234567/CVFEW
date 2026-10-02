"""检查 markdown 的行内 $ 数学定界符配平（排除代码块与 $$ 显示块）。"""
import re, sys

for p in sys.argv[1:]:
    s = open(p, encoding="utf8").read()
    s2 = re.sub(r"```.*?```", "", s, flags=re.S)
    s2 = re.sub(r"\$\$.*?\$\$", "", s2, flags=re.S)
    n = len(re.findall(r"(?<!\\)\$", s2))
    status = "BALANCED" if n % 2 == 0 else "*** UNBALANCED ***"
    print(f"{p}: inline-$ count={n} {status}")
