# Teammate quickstart

1. Unzip the repository and install it from the repository root as described in README.md.
2. Copy the shared raw/aligned/cache/QA artifacts, or point the YAML paths to their existing locations. Keep the episode manifest unchanged.
3. Select `configs/teammate_qwen25_05b.yaml` or `configs/teammate_llama32_1b.yaml` and give `experiment` a unique name.
4. Fetch the model and shared encoders, check the dataset, then launch the three tracks.

```bash
python -m omnibench --config configs/teammate_llama32_1b.yaml fetch --encoders
python -m omnibench --config configs/teammate_llama32_1b.yaml dataset check
python -m omnibench --config configs/teammate_llama32_1b.yaml run --devices 1 --tracks action qa joint_action
```

Llama requires your own authorized Hugging Face access. Authenticate locally using the Hugging Face CLI or standard token environment; do not put credentials in the repository. If weights are already available, set the model entry's `path` and skip fetching.

For limited RAM, set `data.episode_cache_size: 4`. For limited accelerator memory, set EE/QA micro batches and validation/evaluation batches to 1 and enable gradient checkpointing. Keep global batches unchanged for comparable optimization. Full-episode QA has a longer prefix and may require more memory than Action-only. Compatibility with a small CPU fixture is verified; a complete Colab-sized production run is not claimed here.

When running in a notebook, execute the same shell commands with the notebook environment's Python after mounting your data location. The package supports Linux/Colab paths through YAML; no machine directory needs to be edited inside Python code.

A useful Codex instruction is:

> Read AGENTS.md and docs/ADDING_MODELS.md. Use my selected YAML and existing data paths. Run the three public tracks with the specified device count, retain validation-selected checkpoints, and export all held-out split metrics. Do not change the frozen episode membership or use test success for selection.

Return the selected checkpoint sidecars, resolved configuration and results directory to the team. Large runtime artifacts are ignored by Git. No dataset or model weights should be committed.
