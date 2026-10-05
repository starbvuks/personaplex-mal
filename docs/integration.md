# Integration

The adapter expects hidden states shaped `[1, time, host_dim]` and a host exposing `lm.transformer.layers`. `install_branches` adds forward pre-hooks without editing the backbone. Remove the returned handles when finished.

```python
import torch
from mal import MALBranch, install_branches

# lm is your already-loaded compatible PersonaPlex/Moshi language model.
lm.requires_grad_(False)
host_dim = lm.transformer.layers[0].self_attn.embed_dim
branches = torch.nn.ModuleList([
    MALBranch(host_dim, memory_dim=128, persistent_tokens=8)
    for _ in range(4)
]).to(next(lm.parameters()).device)
for branch, index in zip(branches, [7, 15, 23, 31]):
    branch.layer_index = index

handles = install_branches(lm, branches, write=True)
try:
    # Run the host on history frames using its own streaming API.
    # Start each independent dialogue with branch.begin_episode(reset=True).
    pass
finally:
    for handle in handles:
        handle.remove()
```

After the history pass, call `freeze_state()` on every branch and install read-only hooks with `write=False` for query frames. A frozen branch rejects writes. Start a fresh set of branches for a new writable episode; freezing disables parameter gradients.

The zero-based placement above is layers 8, 16, 24 and 32 in a 32-layer host. Confirm the host layout and revision first. The hook tests use a small PyTorch host fixture; this public package does not include a complete native speech runner, codecs, trained adapters or backbone weights. Its quickstart does not validate native speech generation.

Each default gate starts at zero, so an untrained adapter changes no host output. Changing a gate manually can test the path; it does not demonstrate useful learned memory. Host activations and branch parameters also need compatible devices and dtypes. Use float32 branches unless you have checked your mixed-precision policy.

The optional KV path wraps a private Moshi attention method and is version-sensitive. It uses sparse slot writes, not the dense neural-memory gradient rule. Its synthetic unit tests do not establish compatibility with your installed speech model. Keep it disabled for a pass-through baseline: when enabled, even a zero gate adds an attention position and can change attention normalization. The additive branch's zero-gate identity guarantee does not apply to this wrapper.
