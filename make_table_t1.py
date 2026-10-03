"""T1 主表重生成：单一数据源 = B20_grid.json（cls_marginal 修复后 + B21 joint κ profile）。

旧 T1 由修复前的 run_b5.py 产物生成：k=1 行的 phasemap_ml 列（≈20.1，随机水平）
是 ρ≤700 截断引起二次项相消 bug 的陈旧值；k=5 行 phML 受 B21 joint profile 影响
≤0.04pt，其余 13 方法列与 B5_grid 逐位一致（B20 复用同一种子方案与 episode 数据）。

输出：04_results/tables/T1_synthetic_main.md（覆盖旧表，旧表备份 .bak_a04）
"""
import json, os, shutil
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLS = ["euclid", "cosine", "hermitian", "orbital", "tta_euclid", "circcoord",
        "whiten", "drop_agc", "phasemap", "phasemap_ml", "phasemap_w", "oracle"]
SIGMAS = [0.0, np.pi / 12, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]


def main():
    d = json.load(open(os.path.join(BASE, "04_results", "logs", "B20_grid.json")))
    rows = d["records"]

    def get(k, st):
        for r in rows:
            if (r["p"] == 64 and abs(r["rho"] - 0.3) < 1e-9 and r["k"] == k
                    and abs(r["sigma_th"] - st) < 1e-3 and r["fade_q"] == 0.0
                    and r["tag"] == "main"):
                return r
        raise KeyError(f"missing k={k} st={st}")

    lines = ["# T1 合成主网格（p=64, rho=0.3, fade=0, 2000 episodes/格, 5-way）", "",
             f"数据源：`04_results/logs/B20_grid.json`（168 条件 × 21 方法；"
             "cls_marginal 数值稳定重写后 + B21 joint κ profile 的最终配置；"
             "与 B5_grid.json 共用同一种子方案，euclid/orbital 等 13 列逐位一致）", "",
             "| sigma_th | shot | " + " | ".join(COLS) + " |",
             "|---|---|" + "---|" * len(COLS)]
    for k in (5, 1):
        for st in SIGMAS:
            r = get(k, st)
            cells = [f"{100 * r['acc'][m]:.1f}" if m in r["acc"] else "—" for m in COLS]
            lines.append(f"| {st:.2f} | {k} | " + " | ".join(cells) + " |")

    lines += ["", "## 与旧表（B5_grid 产物）的差异（A-04 审计项）", "",
              "- k=1 行 `phasemap` / `phasemap_ml` / `phasemap_w`：旧值 ≈20.1（随机水平）"
              "→ 新值 ≈41.6（=orbital）。原因：修复前 `cls_marginal` 把 ρ 截断到 700，"
              "低 σ²（合成 σ²≈0.07）下 1/σ² 量级二次项失去相消，k=1 退化到随机"
              "（`06_review/audit_2026-10-02.md` [S2]；`verify_cls_marginal.py` 旧实现"
              "有效区回归 100% 一致，故 k=5 与真实 IQ 数字不受影响）。",
              "- k=5 行 `phasemap_ml`：+0.03~+0.36pt（B21 joint profile 相对 plug 的"
              "口径变化，B21 已预注册裁定）；其余 13 方法列逐位不变。",
              "- 论文正文仅引用 k=5 结论；A-02 的独立选择口径列见 "
              "`04_results/tables/T24_a02_independent_selection.md`。"]

    out = os.path.join(BASE, "04_results", "tables", "T1_synthetic_main.md")
    if os.path.exists(out):
        shutil.copy(out, out + ".bak_a04")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("T1 regenerated from B20_grid ->", out)


if __name__ == "__main__":
    main()
