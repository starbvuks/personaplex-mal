import copy
import torch
import pytest
from mal import MALBranch, FastState, update_memory, read_memory, save_checkpoint, load_checkpoint


def test_update_matches_autograd_and_forgetting_momentum():
    torch.manual_seed(3)
    w1 = torch.randn(3, 2, requires_grad=True)
    w2 = torch.randn(2, 3, requires_grad=True)
    u1, u2 = torch.randn_like(w1), torch.randn_like(w2)
    k, v = torch.randn(2), torch.randn(2)
    state = FastState(w1, w2, u1, u2, 4)
    loss = .5 * (read_memory(state, k) - v).square().sum()
    g1, g2 = torch.autograd.grad(loss, (w1, w2), create_graph=True)
    result = update_memory(state, k, v, .2, .7, .03)
    for out, w, u, g in zip((result.w1,result.w2),(w1,w2),(u1,u2),(g1,g2)):
        torch.testing.assert_close(out, .8*w + .7*u - .03*g)
    assert result.writes == 5
    result.w1.sum().backward()
    assert w1.grad is not None


def test_read_before_write_and_chunk_equivalence():
    torch.manual_seed(4)
    a = MALBranch(6, 3, 2)
    b = copy.deepcopy(a)
    a.alpha.data.fill_(.5)
    b.alpha.data.fill_(.5)
    x = torch.randn(1, 4, 6)
    a.begin_episode(); b.begin_episode()
    whole = a(x, write=True)
    chunks = torch.cat([b(x[:,i:i+1], write=True) for i in range(4)], dim=1)
    torch.testing.assert_close(whole,chunks)
    torch.testing.assert_close(a.fast.w1,b.fast.w1)
    # Current value cannot affect its own read, only subsequent reads.
    c = MALBranch(6,3,0); d = copy.deepcopy(c)
    c.alpha.data.fill_(.5); d.alpha.data.fill_(.5)
    d.value.weight.data.mul_(5)
    torch.testing.assert_close(c(x[:,:1],write=True), d(x[:,:1],write=True))
    assert not torch.equal(c.fast.w2,d.fast.w2)


def test_zero_gate_identity_and_frozen_immutability():
    branch=MALBranch(6,3,2)
    x=torch.randn(1,3,6)
    assert torch.equal(branch(x,write=True),x)
    branch.freeze_state()
    before=branch.fast.payload()
    params={k:v.clone() for k,v in branch.state_dict().items()}
    branch(x)
    for k,v in before.items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,branch.fast.payload()[k])
        else: assert v==branch.fast.payload()[k]
    for k,v in params.items(): assert torch.equal(v,branch.state_dict()[k])
    with pytest.raises(RuntimeError): branch(x,write=True)


def test_checkpoint_roundtrip_and_integrity(tmp_path):
    a=torch.nn.ModuleList([MALBranch(6,3,2)])
    a[0](torch.randn(1,4,6),write=True); a[0].freeze_state()
    path=tmp_path/'state.pt'
    save_checkpoint(path,a,{'model_revision':'test','history_sha256':'abc'})
    b,meta=load_checkpoint(path,expected_revision='test')
    x=torch.randn(1,2,6)
    assert torch.equal(a[0](x), b[0](x))
    assert b[0].frozen and meta['history_sha256']=='abc'
    data=torch.load(path,weights_only=True)
    data['interface']['0.alpha'].add_(1)
    torch.save(data,path)
    with pytest.raises(ValueError,match='hash'): load_checkpoint(path)
    torch.save({'state_dict':{}},path)
    with pytest.raises(ValueError,match='format'): load_checkpoint(path)


def test_feature_rms_norm_backward_is_finite_for_zero_norm_frame():
    """Regression: a frame whose mean square is exactly 0 made the old
    sqrt().clamp_min() RMS norm return NaN gradient (0 * inf in SqrtBackward0).
    Clamping the variance before sqrt keeps the forward identical and the
    derivative finite. The real MPS backbone hit this on a bf16 zero frame."""
    torch.manual_seed(0)
    b=MALBranch(16,8,2,policy={'feature_rms_norm':True})
    b.alpha.data.fill_(.5)
    x=torch.randn(1,5,16)
    x[0,2]=0.0                    # zero-RMS frame
    x.requires_grad_(True)
    out=b(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in b.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_outer_objective_reaches_writer_and_data_dependent_rates():
    torch.manual_seed(19)
    b=MALBranch(6,3,2)
    b.alpha.data.fill_(.4)
    # A training update can make rates input-dependent; default untrained weights are zero.
    with torch.no_grad(): b.rates.weight[0].fill_(.1)
    r1=b.rates(torch.ones(3)).sigmoid()
    r2=b.rates(-torch.ones(3)).sigmoid()
    assert not torch.equal(r1,r2)
    b(torch.randn(1,4,6),write=True)
    query_loss=b(torch.randn(1,2,6)).square().sum()
    query_loss.backward()
    for parameter in [b.initial_w1,b.initial_w2,b.key.weight,b.value.weight,b.rates.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
