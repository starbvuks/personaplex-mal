# personaplex-mal

Test-time memory branches for PersonaPlex’s Moshi transformer. Experimental.

A small MLP writes to its own fast weights while the host stays frozen. Reads feed back through a zero-initialized residual gate. Episode state can be frozen, hashed and saved separately from the backbone.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python examples/episode.py
pytest -q
```

The example uses synthetic hidden states on CPU. No model download, microphone or training job.

`mal.py` contains the memory update and temporal-layer hooks. `memory_policy.py` controls writes and residual bounds. `kv_read.py` is an optional sparse-slot attention experiment; it requires a compatible Moshi installation for its attention wrapper.

The mechanics have tests. Successful episodic recall in PersonaPlex is **not established**. This is an independent implementation, not an official adapter or a reproduction of either paper’s results.

[Integration](docs/integration.md) · [Experiments](docs/experiments.md) · [Titans](https://arxiv.org/abs/2501.00663) · [Titans-as-a-Layer](https://arxiv.org/abs/2606.08573)
