# Included measurements

- `selected_results.json`: selected completed configurations for the three backbones, including Qwen3.5-4B Action-only and completed Joint-to-Action. Each profile includes split/task execution counts, checkpoint hashes and separately identified QA-only results.
- `qwen35_4b_joint_action.json`: the completed final 4B checkpoint, with actual assisted execution, offline EE, same-checkpoint QA retention, feature interventions and inference-time modality ablations.

The final 4B Joint-to-Action execution result is 132/209 (63.2%): 67/69 IID, 39/70 compositional and 26/70 visual OOD. All 209 scenarios were evaluated. Per-task results and confidence intervals are retained. These are aggregate measurements; observations, gold-question rows, model weights and training logs are not included.

Read `docs/RESULTS.md` for training provenance, metric definitions and comparison limits. The README QA-only column and execution column are separate checkpoints, whereas the retention metrics in the detailed final-checkpoint JSON use the execution checkpoint itself.
