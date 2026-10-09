"""Synthetic checks of the full-state cap; no model or recall evaluation."""
import torch
import pytest
from mal import FastState, MALBranch


@pytest.mark.parametrize('cap', [0., .05, 100.])
def test_full_state_cap_includes_forgetting_and_scales_momentum(cap):
    branch = MALBranch(2, 2, 0, policy={
        'rates_mode': 'manual', 'forget': .8, 'momentum': .5,
        'learning_rate': 0., 'state_update_max_norm': cap,
    }).double()
    # No learning impulse: this isolates forgetting and carried momentum.
    before = FastState(torch.ones(2, 2, dtype=torch.float64),
                       torch.full((2, 2), 2., dtype=torch.float64),
                       torch.full((2, 2), .2, dtype=torch.float64),
                       torch.full((2, 2), -.4, dtype=torch.float64), 3)
    branch.fast = before
    branch(torch.ones(1, 1, 2, dtype=torch.float64), write=True)
    after = branch.fast
    impulse1, impulse2 = .5 * before.u1, .5 * before.u2
    change1 = -.8 * before.w1 + impulse1
    change2 = -.8 * before.w2 + impulse2
    norm = torch.cat([change1.flatten(), change2.flatten()]).norm()
    scale = min(1., cap / norm.item()) if cap else 1.
    torch.testing.assert_close(after.w1, before.w1 + scale * change1)
    torch.testing.assert_close(after.w2, before.w2 + scale * change2)
    torch.testing.assert_close(after.u1, scale * impulse1)
    torch.testing.assert_close(after.u2, scale * impulse2)
    assert after.writes == 4
    actual = torch.cat([(after.w1 - before.w1).flatten(),
                        (after.w2 - before.w2).flatten()]).norm().item()
    if cap == .05:
        # The older momentum-only cap would leave a much larger forgetting step.
        assert norm.item() > cap * 10
        assert actual == pytest.approx(cap)
    else:
        assert actual == pytest.approx(norm.item())


def test_full_state_cap_bounds_each_prefix_and_history_write():
    torch.manual_seed(23)
    cap = .01
    branch = MALBranch(6, 3, 2, policy={'state_update_max_norm': cap}).double()
    writes = []

    def observe(before, after, info):
        change = torch.cat([(after.w1 - before.w1).flatten(),
                            (after.w2 - before.w2).flatten()]).norm().item()
        assert change <= cap + 1e-12
        assert after.writes == before.writes + 1
        writes.append((branch.write_position, change, info['state_update_scale'].item()))

    branch.write_observer = observe
    branch(torch.randn(1, 3, 6, dtype=torch.float64) * 100, write=True)
    assert [position for position, _, _ in writes] == [-1, -2, 0, 1, 2]
    assert branch.fast.writes == 5
    # Ensure the bound was exercised, not merely checked on tiny natural updates.
    assert any(scale < 1. for _, _, scale in writes)
