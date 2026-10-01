import torch, sys
sys.path.insert(0, "/tmp/cvfe_work/code")
from cvfe.nets_b19 import ComplexAMC2
from cvfe.nets_b17 import unit_norm
from run_b19 import l1pca_proto

for pooling in ("power", "nopool", "autocorr"):
    m = ComplexAMC2(init="hann", pooling=pooling, ch=(48, 96, 192, 48)).cuda()
    z = torch.randn(4, 128, dtype=torch.complex64).cuda()
    e = m(z)
    print(pooling, "out dtype:", e.dtype, "shape:", tuple(e.shape))
    es = unit_norm(e.reshape(1, 2, 2, -1))
    print("  after unit_norm:", es.dtype)
    try:
        mu = unit_norm(l1pca_proto(es, 5))
        ip = torch.einsum("bmc,bnc->bmn", torch.randn(1, 3, 2).cuda(), mu)
        print("  l1pca ok, mu:", mu.dtype, tuple(mu.shape))
    except Exception as ex:
        print("  l1pca FAIL:", ex)
