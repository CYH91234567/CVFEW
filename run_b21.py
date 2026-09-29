"""B21 v3：κ̂ 轮廓 EM 变体对比（PREREG_B21 修正案 v3）。

变体：plug / joint(argmax) / joint_c(min_elig,95%集下界) / joint_os(null_first,单侧保守)。
U1 σ_θ=0 恢复：σ_θ̂ 中位数 ≤ 0.15
U2 σ_θ=π：σ_θ̂ 中位数 ≥ 1.4；phML 不比 plug 差 >0.5pt
U3 σ_θ=π/3 恢复：σ_θ̂ 中位数 ∈ [0.4, 1.5]
U4 非劣性：σ_θ=0, p=16, r=4, 600 epi：phML−euclid bootstrap 95% CI 下界 ≥ −0.5pt
U5 无回归：p=16, r=2, σ_θ=π/3, 600 epi：phML−phML(plug) ≥ −0.5pt
U6 优势守卫（修正格）：p=16, r=4, σ_θ=π/6, 600 epi：phML − max(端点) ≥ +0.5pt
U7（v3）π 端点损失：p=16, r=4, σ_θ=π：phML − orbital ≥ −0.9pt
决策 v3：过 {U4,U6,U7} 者中取 π 损失最小；皆败 → plug。
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth, estim as E

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILES = {
    "plug": dict(kappa_profile="plug"),
    "joint": dict(kappa_profile="joint"),
    "joint_c": dict(kappa_profile="joint", profile_select="min_elig", profile_delta=1.92),
    "joint_os": dict(kappa_profile="joint", profile_select="null_first", profile_delta=1.92),
}


def boot_ci_low(d, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5))


def cell(p, r, st, n_epi, seed0=1000, K=5, k=5, m=15, rho=0.3, seeds=3):
    noise = float(np.sqrt(r / p))
    mu = synth.make_prototypes(K, p, rho, np.random.RandomState(7 + p))
    acc = {"euclid": [], "orbital": []}
    acc.update({f"phML_{tag}": [] for tag in PROFILES})
    sthat = {tag: [] for tag in PROFILES}
    for sd in range(seeds):
        Zs, Zq, yq = synth.batch_episodes(mu, K, k, m, st, noise, n_epi,
                                          seed=seed0 * (sd + 1) + p)
        acc["euclid"].extend((E.run_method("euclid", Zs, Zq)[0] == yq).mean(1).tolist())
        acc["orbital"].extend((E.run_method("orbital", Zs, Zq)[0] == yq).mean(1).tolist())
        for tag, kw in PROFILES.items():
            mu_h, aux = E.phasemap_em(Zs, **kw)
            acc[f"phML_{tag}"].extend(
                ((E.cls_marginal(Zq, mu_h, aux) == yq).mean(1)).tolist())
            sthat[tag].extend(aux["sigma_th"][:, 0].tolist())
    return acc, sthat


def m100(v):
    return 100 * float(np.mean(v))


def main():
    t0 = time.time()
    res = {"cells": {}, "checks": {}}

    # U1 + U4：σ_θ=0
    acc, sth = cell(16, 4, 0.0, 200)
    res["cells"]["st0_r4"] = {"acc": {k: m100(v) for k, v in acc.items()},
                              "st_hat_median": {t: float(np.median(sth[t])) for t in PROFILES}}
    for tag in ("joint", "joint_c", "joint_os"):
        res["checks"][f"U1_st_hat<=0.15 [{tag}]"] = {
            "value": res["cells"]["st0_r4"]["st_hat_median"][tag],
            "pass": bool(res["cells"]["st0_r4"]["st_hat_median"][tag] <= 0.15)}
    for tag in ("joint", "joint_c", "joint_os"):
        d = np.array(acc[f"phML_{tag}"]) - np.array(acc["euclid"])
        lo = boot_ci_low(d)
        res["checks"][f"U4_noninf>=-0.5pt [{tag}]"] = {
            "mean_diff_pt": 100 * float(d.mean()), "ci_low_pt": 100 * lo,
            "pass": bool(100 * lo >= -0.5)}

    # U2 + U7：σ_θ=π
    acc, sth = cell(16, 4, np.pi, 200)
    res["cells"]["stpi_r4"] = {"acc": {k: m100(v) for k, v in acc.items()},
                               "st_hat_median": {t: float(np.median(sth[t])) for t in PROFILES}}
    res["checks"]["U2_st_hat>=1.4"] = {
        "value": res["cells"]["stpi_r4"]["st_hat_median"]["joint"],
        "pass": bool(res["cells"]["stpi_r4"]["st_hat_median"]["joint"] >= 1.4)}
    for tag in ("joint", "joint_c", "joint_os"):
        loss = m100(acc[f"phML_{tag}"]) - m100(acc["orbital"])
        res["checks"][f"U7_pi_endpoint_loss>=-0.9pt [{tag}]"] = {
            "phML": m100(acc[f"phML_{tag}"]), "orbital": m100(acc["orbital"]),
            "loss_pt": loss, "pass": bool(loss >= -0.9)}
    res["checks"]["U2_phML_joint_not_worse>0.5pt_vs_plug"] = {
        "joint": m100(acc["phML_joint"]), "plug": m100(acc["phML_plug"]),
        "pass": bool(m100(acc["phML_joint"]) >= m100(acc["phML_plug"]) - 0.5)}

    # U3：σ_θ=π/3
    acc, sth = cell(16, 4, np.pi / 3, 200)
    res["cells"]["stpi3_r4"] = {"acc": {k: m100(v) for k, v in acc.items()},
                                "st_hat_median": {t: float(np.median(sth[t])) for t in PROFILES}}
    res["checks"]["U3_st_hat_in_[0.4,1.5] [joint]"] = {
        "value": res["cells"]["stpi3_r4"]["st_hat_median"]["joint"],
        "pass": bool(0.4 <= res["cells"]["stpi3_r4"]["st_hat_median"]["joint"] <= 1.5)}

    # U5：无回归 p=16, r=2, σ_θ=π/3
    acc, sth = cell(16, 2, np.pi / 3, 200)
    res["cells"]["stpi3_r2"] = {"acc": {k: m100(v) for k, v in acc.items()}}
    for tag in ("joint", "joint_c", "joint_os"):
        dj = m100(acc[f"phML_{tag}"]); dp = m100(acc["phML_plug"])
        res["checks"][f"U5_vs_plug>=-0.5pt [{tag}]"] = {
            "phML": dj, "plug": dp, "pass": bool(dj - dp >= -0.5)}

    # U6（修正格）：p=16, r=4, σ_θ=π/6
    acc, sth = cell(16, 4, np.pi / 6, 200)
    res["cells"]["stpi6_r4"] = {"acc": {k: m100(v) for k, v in acc.items()},
                                "st_hat_median": {t: float(np.median(sth[t])) for t in PROFILES}}
    for tag in ("joint", "joint_c", "joint_os"):
        adv = m100(acc[f"phML_{tag}"]) - max(m100(acc["euclid"]), m100(acc["orbital"]))
        res["checks"][f"U6_advantage>=+0.5pt [{tag}]"] = {
            "phML": m100(acc[f"phML_{tag}"]),
            "best_endpoint": max(m100(acc["euclid"]), m100(acc["orbital"])),
            "advantage_pt": adv, "pass": bool(adv >= 0.5)}

    res["elapsed_min"] = round((time.time() - t0) / 60, 1)
    out = os.path.join(BASE, "04_results", "logs", "B21_kappa_profile.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["checks"], indent=1))

    def passes(tag):
        keys = [f"U4_noninf>=-0.5pt [{tag}]", f"U6_advantage>=+0.5pt [{tag}]",
                f"U7_pi_endpoint_loss>=-0.9pt [{tag}]"]
        return all(res["checks"][k]["pass"] for k in keys)

    winners = [t for t in ("joint", "joint_c", "joint_os") if passes(t)]
    if winners:
        verdict = min(winners, key=lambda t: res["checks"][
            f"U7_pi_endpoint_loss>=-0.9pt [{t}]"]["loss_pt"])
    else:
        verdict = "plug"
    print(f"B21 DECISION v3: default profile = {verdict} "
          f"({res['elapsed_min']} min) -> {out}")


if __name__ == "__main__":
    main()
