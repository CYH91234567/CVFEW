"""B34 分析：第二数据集（RML2016.10b）上论文 A 真实数据三主张的复现判定。

主张与 10a 对照值（T2/T5 口径）：
  P1 Thm1 实数据验证：PhaseMAP/轨道注入精确平坦（10a: 0.0pt）vs 欧氏下降（10a: −12.2pt）
  P2 SOTA 相位免疫跨架构家族：CNN/CLDNN 注入变化（10a: ≤1.2pt）
  P3 零训练 vs 监督差距量级（10a: 31.5pt）
输出 T22_b34.md。
"""
import json, os
from collections import defaultdict
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(BASE, "04_results", "logs")

# 10a 对照值（T5 重跑后）
REF_10A = {"euclid": 44.1, "orbital": 56.9, "phasemap_ml": 56.9,
           "sota_protonet_cnn": 88.1, "sota_protonet_cldnn": 85.0}


def main():
    d = json.load(open(os.path.join(LOG, "B34_b10b.json")))
    det = defaultdict(lambda: {"all": [], "none": [], "pi2": []})
    sota = defaultdict(lambda: {"all": [], "none": [], "pi2": []})
    for r in d:
        for m, v in r["acc"].items():
            if r["split"] == "all10":
                det[m]["all"].append(100 * v)
                if r["inject_mode"] == "none":
                    det[m]["none"].append(100 * v)
                elif r["inject_mode"] == "global" and abs(r["inject_strength"] - np.pi / 2) < 1e-6:
                    det[m]["pi2"].append(100 * v)
            else:
                sota[m]["all"].append(100 * v)
                if r["inject_mode"] == "none":
                    sota[m]["none"].append(100 * v)
                elif r["inject_mode"] == "global" and abs(r["inject_strength"] - np.pi / 2) < 1e-6:
                    sota[m]["pi2"].append(100 * v)

    def stats(d_, m):
        e = d_[m]
        if not e["all"]:
            return None
        return (float(np.mean(e["all"])),
                float(np.mean(e["none"]) - np.mean(e["pi2"])) if e["none"] and e["pi2"] else float("nan"))

    tab = ["# T22 — B34 第二数据集（RML2016.10b）验证：论文 A 真实数据三主张的复现", "",
           "数据：RML2016.10b（10 类，镜像重建，SNR{0,6,12,18}，z(240000,128)）。",
           "确定性方法用全 10 类 5-way 5-shot（对齐 T2 口径）；SOTA 用固定分层 5/2/3 划分、",
           "test 类 3-way 5-shot、b=4000 plain（对齐 T5 口径）。每格 600/400 episodes，注入 {自然,π/6,π/2}。", "",
           "## 确定性方法（Thm1 实数据验证；括号内为 10a 对照值）", "",
           "| 方法 | 10b 平均acc | 10b π/2 注入drop | 10a 对照 acc(drop) |",
           "|---|---|---|---|"]
    labels = [("euclid", "欧氏（零训练）"), ("orbital", "轨道（零训练）"),
              ("phasemap_ml", "PhaseMAP-ML（零训练）")]
    det_rows = {}
    for m, lab in labels:
        s = stats(det, m)
        if s:
            ref = REF_10A.get(m)
            rtxt = f"{ref:.1f}(−12.2)" if m == "euclid" else f"{ref:.1f}(0.0)" if ref else "—"
            drop_s = "—" if np.isnan(s[1]) else f"{s[1]:+.1f}"
            tab.append(f"| {lab} | {s[0]:.1f} | {drop_s} | {rtxt} |")
            det_rows[m] = s
    tab += ["", "## 监督 SOTA（跨架构家族注入免疫；括号内为 10a 对照值）", "",
            "| 方法 | 10b 平均acc | 10b π/2 注入变化 | 10a 对照 acc(变化) |",
            "|---|---|---|---|"]
    sota_rows = {}
    for m, lab in [("sota_protonet_cnn", "ProtoNet-CNN (b=4000)"),
                   ("sota_protonet_cldnn", "ProtoNet-CLDNN (b=4000)")]:
        s = stats(sota, m)
        if s:
            ref = REF_10A.get(m, 0.0)
            rtxt = f"{ref:.1f}(≤1.2)"
            drop_s = "—" if np.isnan(s[1]) else f"{s[1]:+.1f}"
            tab.append(f"| {lab} | {s[0]:.1f} | {drop_s} | {rtxt} |")
            sota_rows[m] = s

    # 三主张判定
    tab += ["", "## 三主张复现判定（机器生成，阈值：平坦=|drop|≤0.5pt；免疫=|变化|≤2pt）", ""]
    verdict = {}
    if "orbital" in det_rows and "euclid" in det_rows:
        flat_ok = abs(det_rows["orbital"][1]) <= 0.5 and abs(det_rows["phasemap_ml"][1]) <= 0.5
        eucl_drop = det_rows["euclid"][1]
        tab.append(f"- **P1（Thm1 实数据验证）**：轨道/PhaseMAP π/2 drop "
                   f"{det_rows['orbital'][1]:+.2f}/{det_rows['phasemap_ml'][1]:+.2f} pt（平坦判据"
                   f"{'通过' if flat_ok else '未通过'}）；欧氏 drop {eucl_drop:+.1f} pt"
                   f"（10a 为 −12.2）→ {'复现' if flat_ok and euclid_drop < -5 else '部分复现/未复现'}")
        verdict["P1_thm1_repro"] = bool(flat_ok and euclid_drop < -5)
    if "sota_protonet_cnn" in sota_rows:
        imm = abs(sota_rows["sota_protonet_cnn"][1]) <= 2 and (
            "sota_protonet_cldnn" not in sota_rows or abs(sota_rows["sota_protonet_cldnn"][1]) <= 2)
        tab.append(f"- **P2（SOTA 跨架构家族注入免疫）**：CNN/CLDNN π/2 变化 "
                   f"{sota_rows['sota_protonet_cnn'][1]:+.1f}/"
                   + (f"{sota_rows['sota_protonet_cldnn'][1]:+.1f}" if "sota_protonet_cldnn" in sota_rows else "—")
                   + f" pt → {'复现' if imm else '未复现'}")
        verdict["P2_sota_immune_repro"] = bool(imm)
    if "sota_protonet_cnn" in sota_rows and "phasemap_ml" in det_rows:
        gap = sota_rows["sota_protonet_cnn"][0] - det_rows["phasemap_ml"][0]
        tab.append(f"- **P3（零训练 vs 监督差距量级）**：10b 差距 {gap:.1f} pt"
                   f"（10a 为 31.5）→ 量级{'一致' if 20 < gap < 45 else '明显不同，需说明'}")
        verdict["P3_gap_10b_pt"] = round(gap, 1)
    verdict["dataset"] = "RML2016.10b (rebuilt from mirror)"

    out_tab = os.path.join(BASE, "04_results", "tables", "T22_b34.md")
    with open(out_tab, "w", encoding="utf-8") as f:
        f.write("\n".join(tab))
    json.dump(verdict, open(os.path.join(LOG, "B34_verdict.json"), "w"), indent=1)
    print("\n".join(tab))
    print(f"\n-> {out_tab}; B34_verdict.json")


if __name__ == "__main__":
    main()
