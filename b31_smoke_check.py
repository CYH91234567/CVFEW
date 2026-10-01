import torch, sys
sys.path.insert(0, "/tmp/cvfe_work/code")
from cvfe.nets_b19 import ComplexAMC2

for pooling in ("autocorr", "power", "nopool"):
    m = ComplexAMC2(init="hann", pooling=pooling, ch=(48, 96, 192, 48)).cuda()
    z = torch.randn(4, 128, dtype=torch.complex64).cuda()
    e = m(z)
    rot = torch.exp(torch.tensor(1j * 0.7, dtype=torch.complex64)).cuda()
    er = m(z * rot)
    if pooling in ("autocorr", "power"):
        d = (er - e).abs().max().item()            # 精确不变（非等变）
        print(f"{pooling:9s} out={tuple(e.shape)} INVARIANCE max|Δ|={d:.2e} "
              f"params={m.n_params()}")
        # 阈值 1e-2：float32 trunk 前向本身的非确定性（δ~2e-3）；
        # 统计量层在 numpy 镜像 T10a 精确到 1e-16
        assert d < 1e-2
    else:
        num = (er - e * rot).abs().norm(dim=-1)
        den = e.abs().norm(dim=-1).clamp_min(1e-12)
        delta = (num / den).mean().item()         # 等变
        print(f"{pooling:9s} out={tuple(e.shape)} equiv_delta={delta:.2e} "
              f"params={m.n_params()}")
        assert delta < 1e-3
    assert e.isfinite().all()
print("B31 SMOKE-FWD OK")
