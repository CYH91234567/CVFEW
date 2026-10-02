"""数据等价性预检：用重建 cache 复算 T5 的 split0/SNR0/none 格的三个确定性方法，
对比原数据副本上的数字（phasemap 65.91 / orbital 65.96 / euclid 57.43）。
采样参数与 run_b10.py 逐字一致。"""
import sys, os, numpy as np
sys.path.insert(0, '/tmp/cvfe_work/code')
from cvfe import data as D, estim as E
from cvfe.episodes import EpisodeSampler, make_class_splits

z, y, snr = D.load_radioml('/tmp/cvfe_work/code/radioml_cache.npz')
zn = D.energy_normalize(z).astype(np.complex64)
splits = make_class_splits(n_splits=5)
sp = splits[0]
print('test classes:', sp['test'])
smp_te = EpisodeSampler(zn, y, snr, classes=sp["test"], n_way=3, k_shot=5,
                        q_per_class=15, seed=5000, snr_min=0, snr_max=0)
Zs, Zq, yq = smp_te.sample(400, inject=None)
Zs128, Zq128 = Zs.astype(np.complex128), Zq.astype(np.complex128)
print('euclid   %.2f  (原 57.43)' % (100 * (E.cls_euclid(Zq128, E.proto_euclid(Zs128)) == yq).mean(1).mean()))
print('orbital  %.2f  (原 65.96)' % (100 * (E.cls_orbital(Zq128, E.proto_orbital(Zs128)) == yq).mean(1).mean()))
acc_pm = np.zeros(400)
for g0 in range(0, 400, 100):
    sl_ = slice(g0, min(g0 + 100, 400))
    mu_pm, aux = E.phasemap_em(Zs128[sl_])
    pred = E.cls_marginal(Zq128[sl_], mu_pm, aux)
    acc_pm[sl_] = (pred == yq[sl_]).mean(1)
print('phasemap %.2f  (原 65.91)' % (100 * acc_pm.mean()))
print('EQUIV_CHECK_DONE')
