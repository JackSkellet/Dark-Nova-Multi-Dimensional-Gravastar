# Problem Definition

## Definitions

- **Quality:** task-specific correctness measured by validation loss, accuracy, functional tests, documentation correctness, or security checks. Text similarity alone is insufficient for coding tasks.
- **Compression:** reduction in total encoded bytes versus a baseline after counting values, metadata, indexes, scales, codebooks, padding, caches, and reconstruction workspace.
- **Runtime memory:** peak RAM or VRAM required for model state, active components, retrieval/indexes, caches, activations, and temporary buffers.
- **Active parameters:** parameters read or used for a specific input, excluding dormant parameters but including router and selected component overhead.
- **Memory traffic:** bytes moved through the critical execution path, including lookup, routing, transfer, reconstruction, and kernel inputs.
- **Latency:** measured p50 and p95 wall-clock time for the complete operation under test.
- **Continual adaptation:** a controlled update to retrieval memory, structured memory, adapter/expert parameters, or a checkpoint after new authorized data arrives.
- **Real time:** updates visible within an interactive session, normally seconds.
- **Near real time:** updates available after automated validation, normally minutes.
- **Periodic consolidation:** infrequent production of a candidate checkpoint from accumulated validated state.
- **Catastrophic forgetting:** degradation on retained prior tasks after adapting to new tasks.
- **Carrier pathway:** a candidate tensor, channel, expert, adapter, feature, or direction hypothesized to carry disproportionate functional importance.
- **Semantic route:** a human-interpretable label for a computational pathway; it is not evidence of causal specialization by itself.
- **Multi-dimensional representation:** a logical parameter object indexed by axes such as layer, route, component, and version; physical compression must still reduce total bytes.
- **Security failure:** unauthorized access, leakage, poisoning, backdoor activation, tampering, rollback failure, or unapproved network/data flow.

## Separated Mechanisms

The project separates pretraining, fine-tuning, post-training compression, runtime routing, storage representation, retrieval-memory updates, adapter updates, and checkpoint consolidation.

