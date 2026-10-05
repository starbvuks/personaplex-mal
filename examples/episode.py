"""Exercise episode state on CPU. This is not a speech or recall evaluation."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mal import MALBranch, save_checkpoint, load_checkpoint

torch.manual_seed(7)
branches = torch.nn.ModuleList([MALBranch(16, 8, 2) for _ in range(4)])
history = torch.randn(1, 5, 16)
query = torch.randn(1, 2, 16)
for branch, layer in zip(branches, [7, 15, 23, 31]):
    branch.layer_index = layer
    # An untrained gate is an exact pass-through while episode state still writes.
    assert torch.equal(branch(history, write=True), history)
    branch.freeze_state()

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "episode.pt"
    save_checkpoint(path, branches, {"model_revision": "synthetic-cpu-fixture"})
    restored, metadata = load_checkpoint(path, expected_revision="synthetic-cpu-fixture")
    for before, after in zip(branches, restored):
        torch.testing.assert_close(before(query), after(query), rtol=0, atol=0)
    print("CPU fixture: zero-gate identity and frozen-state round-trip passed.")
    print("Writes per branch:", [branch.fast.writes for branch in restored])
    print("State SHA-256:", metadata["state_sha256"][0])
    print("No speech model or episodic recall was evaluated.")
