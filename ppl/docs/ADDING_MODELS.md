# Adding a backbone

For Qwen2.5-0.5B and Llama-3.2-1B, the supplied YAML entries already select the correct Hugging Face architecture. Choose the corresponding teammate config; no Python edit should be needed.

For another compatible backbone, add a model registry entry:

```yaml
models:
  my_backbone:
    name: My-Backbone
    hf_id: organization/model-name
    directory: My-Backbone
    family: auto
model: my_backbone
```

An optional `path` overrides the model directory. Local model files must include model configuration, tokenizer and base weights. Paths resolve from the configured project root.

The current adapter requires a causal language model with `.model`, `.lm_head` and `.get_input_embeddings()`. Generic attention adapters target `q_proj` and `v_proj`. Qwen3.5 is handled separately by extracting its language model and adding gated-DeltaNet LoRA targets; its native visual tower is unused. Another architecture with different internals needs a small adapter in `BenchModel`, not merely a new name in YAML.

Keep the same sensory inputs, frozen encoder assets, partition manifest, history window and token budgets when comparing backbones. Check the architecture's dtype and attention behavior. Before a full run, run tests and a small local forward/backward check for finite gradients through encoders, projectors, LoRA and the action head. The existing Qwen2 and Llama fixtures demonstrate this contract without downloading weights.

If batch sizes, training schedules, external encoders or modality budgets change, record the change in the experiment configuration and results. Do not present a changed recipe as an otherwise identical backbone comparison.
