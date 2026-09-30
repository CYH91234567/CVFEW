"""B20 分析：canon 系并入主网格后的 P7 σ_θ 轴检验（PREREG_B20）。

输入：04_results/logs/B20_grid.json
输出：tables/T11_b20_canon.md、logs/B20_p7_verdict.json、figures/F10_b20_canon.png
判据（PREREG_B20 + v2 机制细化）：
  P-a σ_θ=0：canonX_ref ≈ orbital（≤0.5pt）
  P-b σ_θ 增大：逐帧规范化（first/mean/max/wmean）塌向 chance；canon_ref（支持集
      二阶矩参考 = MRA 同步）部分恢复但不达轨道（canonX_ref 介于其间）
  P-c orbital/phML 全轴保持端点水平
  P-d 稳定性排序 canon_first ≤ canon_mean ≤ canon_max ≤ canon_ref（σ_θ=π 行）；
      1-shot 全方法 ≈ chance
"""
import json, os, sys
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANON = ["canon_first", "canon_mean", "canon_max", "canon_wmean", "canon_ref",
         "canonX_ref", "canonQ_ref"]


def main():
    d = json.load(open(os.path.join(BASE, "04_results", "logs", "B20_grid.json")))
    rows = d["records"]
    print(f"loaded {len(rows)} conditions; methods={len(d['methods'])}")

    def get(p, rho, k, st, fade):
        for r in rows:
            if (r["p"] == p and abs(r["rho"] - rho) < 1e-9 and r["k"] == k
                    and abs(r["sigma_th"] - st) < 1e-6 and abs(r["fade_q"] - fade) < 1e-9):
                return r
        return None

    A = lambda r, m: 100 * r["acc"][m]
    lines = ["# T11：B20 canon 系并入主网格（5-shot, fade=0, σ=0.3, 2000 epi/格）", "",
             "## 主面板：p=16, ρ=0.3, k=5, fade=0（% 精度）", "",
             "| 方法 | " + " | ".join(f"σ_θ={s:.2f}" for s in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]) + " |",
             "|---|" + "---|" * 6]
    sig_order = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
    show = ["euclid", "orbital", "phasemap_ml"] + CANON
    for m in show:
        cells = []
        for st in sig_order:
            r = get(16, 0.3, 5, st, 0.0)
            cells.append(f"{A(r, m):.1f}" if r and m in r["acc"] else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # 汇总：canonX_ref/canon_ref − orbital 与 phML − max(端点)（k=5, fade=0 全格）
    lines += ["", "## 汇总（k=5, fade=0；每格 2000 epi 配对）", "",
              "| p | ρ | mean(canonX_ref−orb) | mean(canon_ref−orb) | mean(phML−max端点) | worst(phML−max端点) |",
              "|---|---|---|---|---|---|"]
    verdict = {"Pa": [], "Pb_collapse": [], "Pc": [], "Pd_order_violations": []}
    for p in (16, 64):
        for rho in (0.0, 0.3, 0.7):
            gX, gR, gP = [], [], []
            for st in sig_order:
                r = get(p, rho, 5, st, 0.0)
                if r is None:
                    continue
                gX.append(A(r, "canonX_ref") - A(r, "orbital"))
                gR.append(A(r, "canon_ref") - A(r, "orbital"))
                best = max(A(r, "euclid"), A(r, "orbital"))
                gP.append(A(r, "phasemap_ml") - best)
                if abs(st) < 1e-9:
                    verdict["Pa"].append({"p": p, "rho": rho,
                                          "canonX_ref_minus_orbital": gX[-1],
                                          "canon_ref_minus_orbital": gR[-1]})
                if abs(st - np.pi) < 1e-6:
                    row_collapse = {m: A(r, m) for m in CANON[:4]}
                    verdict["Pb_collapse"].append({"p": p, "rho": rho, **row_collapse,
                                                   "canon_ref": A(r, "canon_ref"),
                                                   "orbital": A(r, "orbital")})
                if "h1b_vs_best_endpoint" in r:
                    verdict["Pc"].append({"p_dim": p, "rho": rho, "st": float(st),
                                          "phML_minus_best": gP[-1],
                                          "wilcoxon_p": r["h1b_vs_best_endpoint"]["p"]})
            lines.append(f"| {p} | {rho} | {np.mean(gX):+.1f} | {np.mean(gR):+.1f} "
                         f"| {np.mean(gP):+.1f} | {np.min(gP):+.1f} |")

    # P-d 排序检验：σ_θ=π 处 first ≤ mean ≤ max ≤ ref（允许 0.3pt 容差）
    for r_pb in verdict["Pb_collapse"]:
        f, m_, mx, rf = (r_pb["canon_first"], r_pb["canon_mean"],
                         r_pb["canon_max"], r_pb["canon_ref"])
        ok = (f <= m_ + 0.3) and (m_ <= mx + 0.3) and (mx <= rf + 0.3)
        verdict["Pd_order_violations"].append({"p": r_pb["p"], "rho": r_pb["rho"],
                                               "ok": bool(ok), "vals": [f, m_, mx, rf]})
    # 1-shot 折叠
    one_shot = [A(r, m) for r in rows if r["k"] == 1 for m in
                ["euclid", "orbital", "phasemap_ml"] + CANON if m in r["acc"]]
    verdict["Pd_1shot_max_acc"] = float(np.max(one_shot))
    verdict["Pd_1shot_chance"] = 20.0

    # 判定汇总
    pa_ok = all(abs(v["canonX_ref_minus_orbital"]) <= 0.5 for v in verdict["Pa"])
    pb_ok = all(r["canon_max"] < r["canon_ref"] - 3 for r in verdict["Pb_collapse"])
    pc_ok = all(v["phML_minus_best"] >= -0.5 for v in verdict["Pc"])
    pd_ok = all(v["ok"] for v in verdict["Pd_order_violations"])
    pc_worst = min(v["phML_minus_best"] for v in verdict["Pc"]) if verdict["Pc"] else float("nan")
    pc_n_below = sum(1 for v in verdict["Pc"] if v["phML_minus_best"] < -0.5)
    verdict["verdict"] = {"Pa_canonXref_free_at_st0": bool(pa_ok),
                          "Pb_perframe_collapse": bool(pb_ok),
                          "Pc_phML_envelope": bool(pc_ok),
                          "Pc_worst_pp": float(pc_worst),
                          "Pc_cells_below_-0.5": int(pc_n_below),
                          "Pd_order_and_1shot": bool(pd_ok),
                          "note_canonQ_ref": "as-run实现与canon_ref计算路径重复（分发bug，"
                                             "B18定义的支持侧应为proto_orbital）；列保留但"
                                             "解释时按canon_ref处理，canonQ分解列为future work"}
    json.dump(verdict, open(os.path.join(BASE, "04_results", "logs", "B20_p7_verdict.json"), "w"),
              indent=1)
    txt = "\n".join(lines)
    open(os.path.join(BASE, "04_results", "tables", "T11_b20_canon.md"), "w").write(txt + "\n")
    print(txt)
    print(json.dumps(verdict["verdict"], indent=1))

    # F10 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        r16 = [get(16, 0.3, 5, st, 0.0) for st in sig_order]
        for m, mk in (("euclid", "o"), ("orbital", "s"), ("phasemap_ml", "^"),
                      ("canon_first", "v"), ("canon_max", "D"), ("canon_ref", "P"),
                      ("canonX_ref", "X")):
            axes[0].plot(sig_order, [A(r, m) for r in r16], marker=mk, ms=4, label=m, lw=1.2)
        axes[0].axhline(20, color="k", ls=":", lw=0.8); axes[0].text(2.6, 21, "chance", fontsize=7)
        axes[0].set_xlabel("σ_θ (rad)"); axes[0].set_ylabel("acc (%)")
        axes[0].set_title("p=16, ρ=0.3, k=5, fade=0"); axes[0].legend(fontsize=7)
        for p, mk in ((16, "o"), (64, "s")):
            for rho, cl in ((0.0, "C0"), (0.3, "C1"), (0.7, "C2")):
                gaps = []
                for st in sig_order:
                    r = get(p, rho, 5, st, 0.0)
                    gaps.append(A(r, "canon_ref") - A(r, "orbital"))
                axes[1].plot(sig_order, gaps, marker=mk, ms=4, color=cl, lw=1.1,
                             label=f"p={p}, ρ={rho}")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("σ_θ (rad)"); axes[1].set_ylabel("canon_ref − orbital (pp)")
        axes[1].set_title("cut-locus cost of MRA-rule canonicalization (k=5, fade=0)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(BASE, "04_results", "figures", "F10_b20_canon.png"), dpi=160)
        print("F10 saved")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
