# Training

## Public tracks

- **Action-only** (`action`): optimize the causal EE XYZ-delta and gripper objective.
- **QA-only** (`qa`): optimize answer-token cross entropy using complete-episode observations.
- **Joint-to-Action** (`joint_action`): learn a joint action/QA representation, select its lowest validation-QA-CE checkpoint, then specialize with the action objective and select the lowest validation-EE-MAE checkpoint. QA is monitored during specialization but supplies no loss at that stage.

The internal joint stage is required to construct Joint-to-Action. It is not exposed as a fourth public experiment or a separate showcase result.

All tracks begin from the same saved random initialization of the trainable encoders, projectors and heads for a given backbone/seed. The pretrained language weights remain frozen except for LoRA. Frozen ResNet18 and Whisper-tiny features are precomputed; modality encoders and projectors remain trainable.

## Sensor and target interface

The 13 branches are named in `omnibench/model.py`. Each branch emits 128-dimensional representations projected to the backbone hidden dimension. Modality/time embeddings and text form a compact left-padded prefix. Projected tokens are normalized to the backbone embedding RMS. EE uses one token per branch; QA uses 16 per branch. EE observes the current frame and at most 15 preceding frames; whole-episode QA observes the complete episode. Instruction speech is an episode-level condition available from the start.

The action head predicts a world-frame XYZ delta in meters and a normalized finger-opening command. It uses fixed orientation, not unrestricted learned 6DoF. The XYZ loss is Smooth L1 after division by 0.025 m, plus 0.25 times binary gripper cross entropy. The evaluator adds the predicted delta to the current EE position to obtain the absolute target. The next observed position is graded separately.

QA training alternates full-observation and isolated-branch views for each selected question. Gold answers are only decoder supervision. Task goals, simulator event logs, future action observations and expert commands are excluded from sensory inputs. Interventions use donor mappings for construction/grading only.

## Default portable recipe

| Setting | Default |
|---|---|
| LoRA | Rank 16, alpha 32, dropout 0 |
| LoRA LR | 0.000025 |
| Encoder/projector/head LR | 0.00005 |
| Optimizer | AdamW, weight decay 0.00001 |
| Gradient clipping | Global norm 1.0 |
| Global EE / QA batches | 256 / 128 |
| EE / QA micro batches per device | 4 / 2 |
| Warmup | 100 updates |
| LR schedule | Cosine to a 0.1 floor over 6,000 updates; floor retained afterward |
| Validation | Full accepted validation references, every 100 updates |
| QA cap / minimum / patience | 4,000 / 800 / 5 validations |
| Action cap / minimum / patience | 8,000 / 2,000 / 8 validations |
| Joint-stage cap / minimum / patience | 4,000 / 800 / 5 validations |

Caps are safety limits, not a requirement to spend every update. Patience resets on validation improvements exceeding the configured threshold (QA CE 0.002; action MAE 0.01 mm). The lowest raw validation score is still preserved even when the improvement does not reset patience. Test success does not select checkpoints or alter training.

Joint loss balancing starts from four training batches per task using shared-projector gradient norms. It periodically adjusts the QA weight based on actual shared training gradients. It does not use held-out labels. The action-specialization stage starts a fresh optimizer; it does not continue joint optimizer moments.

These YAML defaults provide a consistent prospective recipe for teammate runs. Historical showcase checkpoints have the separate training histories documented in `RESULTS.md`; launching these defaults does not reproduce every historical weight bit for bit.

## Launch and device count

```bash
python -m omnibench --config configs/default.yaml --model qwen35_4b run --devices 16 --tracks action qa joint_action
```

The launcher executes tracks sequentially, distributing each training/evaluation stage across all requested visible accelerators. Global batch sizes must divide the device count. Lower micro batches to reduce accelerator memory without changing global batches. Lower `data.episode_cache_size` to reduce host RAM. Configure `runtime.gradient_checkpointing: true` when additional activation-memory savings are needed.

On CUDA, standard visible-device environment variables select devices; on Ascend use the vendor equivalent. `runtime.device` accepts `auto`, `cuda`, `npu`, or `cpu`. `OMNIBENCH_DEVICE` can override it. BF16 is the tested accelerator precision. Devices without BF16 need `runtime.dtype: float32`, which increases memory needs; reduced fixtures on CPU do not establish full-size training feasibility.

## Completed Qwen3.5-4B recipe

`configs/qwen35_4b_complete.yaml` records the completed 16-device setup. It uses global EE batch 256 in Action-only and the internal joint stage, and 512 during action specialization. `training.joint_action.action_global_ee_batch` overrides only the specialization-stage batch. Other configs fall back to `training.global_ee_batch`. The resolved global batch is saved in every checkpoint, and a resume refuses a changed batch.

```bash
python -m omnibench --config configs/qwen35_4b_complete.yaml run --devices 16 --tracks action qa joint_action
```

This configuration captures the completed recipe and stage-specific batch change. It does not promise bit-identical weights across different runtimes. Joint-to-Action finished after 1,300 joint updates and 4,800 action updates, selecting joint update 800 and action update 4,500 on validation. Full results are in RESULTS.md.

## Saved artifacts and restart

Runtime artifacts go under `paths.outputs`:

```text
outputs/checkpoints/<experiment>/<model>/<mode>/seed42/
  best/final.pt
  best/final_weights.json
  best_qa/...
  best_action/...
  latest.pt
  rng_step<N>_rank<R>.pt
  selection.json
  config.json
  completion.json
outputs/logs/<experiment>/<model>/status.json
outputs/results/<experiment>/<model>/<track>/step<N>/...
```

`final.pt` stores trainable parameters only. Loading it also requires the same base model files and shared external encoder assets. The checkpoint sidecar records its SHA-256. `best_qa` and `best_action` exist when their validation objective is monitored. Stage snapshots are retained at each validation by the default launcher.

Rerun the same `run` command after an interruption: completed stages are skipped and unfinished stages resume from `latest.pt`, restoring AdamW moments, random states and sampler position. Keep the device count, batches, dataset manifests and training configuration unchanged for a true continuation. Completed runs are immutable; use a new `experiment` name for a new recipe. A disk budget is necessary because validation snapshots are retained.

Qwen3.5 uses language-layer LoRA including gated-DeltaNet projections. Llama and Qwen2.5 use attention q/v LoRA. The compatibility module computes the Qwen3.5 recurrent algebra using ordinary PyTorch and uses FP32 attention computation where configured by the adapter. Backend kernels can still produce small numerical differences.
