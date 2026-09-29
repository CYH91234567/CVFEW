"""B14：层级κ收缩实验——修复 fade+小σ_θ 劣势，量化先验价值与误设鲁棒性。

格A（已知劣势格）: fade=0.5, σ_θ=π/6, ρ=0.3, p=64, k=5 — 无先验时 phML 落后 euclid 1.04pt
格B（先验误设鲁棒）: fade=0, σ_θ=π, 先验误设为 π/6 — 收缩不应灾难性误导
格C（低维不可辨识）: p=8（对应B6嵌入维数）, σ_θ=π/3, k=5 — 无先验σ̂不可辨识
方法: phML(无先验) / phML+prior(σ₀=π/6, w0∈{2,5,10}) / phML+oracle-prior / 端点参照
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvfe import synth as S
from cvfe import estim as E

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_results"))
EPI = 2000
CHUNK = 125


def acc_of(pred, yq):
    return float((pred == yq).mean())


def run_cell(tag, p, rho, k, st, fade_q, priors, seed=4242):
    rng = np.random.RandomState(99)
    mu = S.make_prototypes(5, p, rho, rng)
    accs = {name: np.zeros(EPI) for name in
            ["euclid", "orbital", "phML", "phML_oracle_prior"] +
            [f"phML_prior{s0:.2f}_w{w0}" for (s0, w0) in priors]}
    done = 0
    while done < EPI:
        n = min(CHUNK, EPI - done)
        Zs, Zq, yq = S.batch_episodes(mu, 5, k, 75, st, 0.3, n, seed=seed + done, fade_q=fade_q)
        accs["euclid"][done:done+n] = (E.cls_euclid(Zq, E.proto_euclid(Zs)) == yq).mean(1)
        accs["orbital"][done:done+n] = (E.cls_orbital(Zq, E.proto_orbital(Zs)) == yq).mean(1)
        mu_pm, aux = E.phasemap_em(Zs)
        accs["phML"][done:done+n] = (E.cls_marginal(Zq, mu_pm, aux) == yq).mean(1)
        mu_o, aux_o = E.phasemap_em(Zs, kappa_prior=(st, 5.0))
        accs["phML_oracle_prior"][done:done+n] = (E.cls_marginal(Zq, mu_o, aux_o) == yq).mean(1)
        for (s0, w0) in priors:
            mu_p, aux_p = E.phasemap_em(Zs, kappa_prior=(s0, w0))
            accs[f"phML_prior{s0:.2f}_w{w0}"][done:done+n] = \
                (E.cls_marginal(Zq, mu_p, aux_p) == yq).mean(1)
        done += n
    out = {"tag": tag, "p": p, "rho": rho, "k": k, "sigma_th": st, "fade_q": fade_q,
           "acc": {m: float(a.mean()) for m, a in accs.items()},
           "acc_se": {m: float(a.std() / np.sqrt(EPI)) for m, a in accs.items()},
           "epi": EPI}
    print(f"[{tag}] euclid={100*out['acc']['euclid']:.1f} orbital={100*out['acc']['orbital']:.1f} "
          f"phML={100*out['acc']['phML']:.1f} " +
          " ".join(f"{m}={100*v:.1f}" for m, v in out["acc"].items() if "prior" in m),
          flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    priors = [(np.pi / 6, 2.0), (np.pi / 6, 5.0), (np.pi / 6, 10.0)]
    rows = []
    rows.append(run_cell("A_fade_small_sigma", 64, 0.3, 5, np.pi / 6, 0.5, priors))
    rows.append(run_cell("B_wrong_prior", 64, 0.3, 5, np.pi, 0.0, priors))
    rows.append(run_cell("C_lowdim", 8, 0.3, 5, np.pi / 3, 0.0, priors))
    json.dump(rows, open(os.path.join(OUT, "logs", "B14_hierarchical_kappa.json"), "w"), indent=1)
    print("B14 DONE")
