# Experiments

This release extracts the adapter mechanics from a local PersonaPlex workbench. It contains no conversation data, audio, checkpoints or evaluation corpus.

The neural memory is a two-layer SiLU MLP. Queries and keys are normalized; values are not. A single-token update differentiates half the summed squared reconstruction error, combines the gradient with momentum, and applies forgetting to both weight matrices. Interface parameters are intended for outer-loop learning; this release supplies no trained interface. Fast weights and momentum are episode state. Reads happen before the current token writes.

Those choices adapt the papers rather than reproduce their exact architectures, tasks or reported results. The original Titans-as-a-Layer speech paper concerns conversational emotion recognition, not PersonaPlex recall.

Tests check analytic gradients against autograd, chunked and whole-sequence equivalence, zero-gate identity, frozen-state immutability, checkpoint integrity and finite backward gradients for zero-norm frames. These are mechanism checks.

A useful next evaluation must compare matched history, swapped history, empty history and gates-off controls on held-out queries. Measure whether the correct answer follows the supplied history; a changed output, nonzero state or lower training loss is insufficient. A previous local experiment exhibited a history-independent bias. Removing the start-frame write removed that bias but did not establish correct recall.

No recall score or latency improvement is claimed here.
