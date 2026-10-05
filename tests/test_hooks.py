from types import SimpleNamespace
import pytest
import torch
from mal import MALBranch, install_branches
from kv_read import KVMemory


def test_sparse_host_hook_changes_only_target_and_removes_cleanly():
    torch.manual_seed(11)
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    host = SimpleNamespace(transformer=SimpleNamespace(layers=layers))
    branch = MALBranch(6, 3, 0)
    branch.layer_index = 1
    branch.alpha.data.fill_(0.5)
    x = torch.randn(1, 2, 6)
    expected = branch(x)
    assert not torch.equal(expected, x)
    handles = install_branches(host, [branch])
    try:
        assert torch.equal(layers[0](x), x)
        torch.testing.assert_close(layers[1](x), expected)
        assert torch.equal(layers[2](x), x)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(layers[1](x), x)


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3], [True]])
def test_invalid_layer_placement_leaves_host_unmodified(indices):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    host = SimpleNamespace(transformer=SimpleNamespace(layers=layers))
    branches = [MALBranch(6, 3, 0) for _ in indices]
    for branch, index in zip(branches, indices):
        branch.layer_index = index
    with pytest.raises(ValueError):
        install_branches(host, branches)
    assert not any(layer._forward_pre_hooks for layer in layers)


def test_sparse_kv_write_zero_gate_and_bank_roundtrip():
    torch.manual_seed(5)
    memory = KVMemory(memory_dim=4, num_slots=8, topk=2, num_heads=2, head_dim=3)
    memory.write(torch.randn(1, 4))
    assert memory.slot_occupancy() == 2
    k, v = memory.read_kv()
    assert k.shape == v.shape == (2, 3)
    assert torch.count_nonzero(k) == torch.count_nonzero(v) == 0
    restored = KVMemory(**memory.config_dict())
    restored.load_bank(memory.bank_dict())
    assert torch.equal(restored.bank, memory.bank)
