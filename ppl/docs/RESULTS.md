# Selected measured results

This snapshot includes selected completed configurations for three backbones. It does not include tuning runs or a standalone joint result. Every held-out split of each included configuration is retained below. Execution and QA use separate weights. Training budgets and initialization histories differ, so these numbers do not establish a controlled model-size ranking.

## Actual assisted execution

| Model | Track | All (209) | IID (69) | Compositional (70) | Visual OOD (70) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B | Joint-to-Action | 130/209 (62.2%) | 68/69 (98.6%) | 39/70 (55.7%) | 23/70 (32.9%) |
| Llama-3.2-3B-Instruct | Action-only | 112/209 (53.6%) | 66/69 (95.7%) | 37/70 (52.9%) | 9/70 (12.9%) |
| Qwen3.5-4B | Action-only | 126/209 (60.3%) | 69/69 (100.0%) | 38/70 (54.3%) | 19/70 (27.1%) |
| Qwen3.5-4B | Joint-to-Action | 132/209 (63.2%) | 67/69 (97.1%) | 39/70 (55.7%) | 26/70 (37.1%) |

All runs use 96 actual predicted actions per scenario. Task goal and required release are checked. Penetration is diagnostic only. These are assisted kinematic simulation results, not physical or real-robot success. Aggregate JSON retains per-task counts, uncertainty intervals and checkpoint hashes.

## QA-only: full-observation encoder-category macro accuracy

| Model | QA phase step | IID | Compositional | Visual OOD |
|---|---:|---:|---:|---:|
| Qwen3.5-2B | 2000 | 67.7% | 58.4% | 53.7% |
| Llama-3.2-3B-Instruct | 4000 | 71.2% | 57.8% | 20.5% |
| Qwen3.5-4B | 800 | 66.6% | 61.4% | 49.4% |

QA counts are 418 IID, 420 compositional and 425 visual OOD items. Macro accuracy averages observed encoder/question categories; it differs from question-weighted micro accuracy. The QA-only checkpoints above are distinct from the execution checkpoints.

## Completed Qwen3.5-4B Joint-to-Action

The joint stage stopped at update 1,300 and selected its best validation-QA checkpoint at update 800. Action specialization stopped at update 4,800 and selected update 4,500 by validation EE MAE (0.250 mm). The action-specialization global batch is 512 (32 per rank across 16 devices); the joint-stage global EE/QA batches remain 256/128. This batch differs from the Action-only comparison and must be treated as part of the recipe.

On the existing 209 frozen execution scenarios, Joint-to-Action succeeds on 132 episodes (63.2%; episode-bootstrap 95% interval 56.9%–69.4%), compared with 126 (60.3%) for Action-only: six additional successes, or 2.9 percentage points. This is a comparison of completed configurations with different optimization histories, not an isolated causal estimate of joint pretraining. These scenarios were used in earlier project diagnosis; they are not a newly collected unseen test set. No execution errors occurred. Validation execution (67/70) is separate from these test results.

| Same final Joint-to-Action checkpoint: QA metric | IID | Compositional | Visual OOD |
|---|---:|---:|---:|
| Full-observation encoder-category macro accuracy | 50.1% | 43.7% | 39.2% |
| Full-observation question-weighted micro accuracy | 42.3% | 36.9% | 26.1% |

These retention results measure the action-specialized checkpoint; they are not the QA-only results above. Detailed per-task execution results, offline EE metrics, per-category QA, feature interventions and inference-time modality removals are in `results/qwen35_4b_joint_action.json`. The completed track contains no standalone joint-result showcase.

## Qwen2.5-0.5B teammate run

Fresh-start run of the three public tracks with `configs/teammate_qwen25_05b_a100.yaml` (seed 42), one NVIDIA
A100-SXM4-40GB per track. Measurements in `results/qwen25_05b_teammate.json`.

| Track | Selected update | All (209) | IID (69) | Compositional (70) | Visual OOD (70) |
|---|---:|---:|---:|---:|---:|
| Action-only | action 5,500 | 116/209 (55.5%) | 67/69 (97.1%) | 38/70 (54.3%) | 11/70 (15.7%) |
| Joint-to-Action | joint 1,700, action 2,400 | 126/209 (60.3%) | 61/69 (88.4%) | 39/70 (55.7%) | 26/70 (37.1%) |

Execution uses Wilson 95% intervals in the JSON (not the episode bootstrap of the historical profiles).

| QA metric (full observation) | IID | Compositional | Visual OOD |
|---|---:|---:|---:|
| QA-only (update 1,800), encoder-category macro | 63.1% | 55.2% | 44.0% |
| QA-only, question-weighted micro | 67.8% | 63.5% | 33.6% |
| Joint-to-Action checkpoint (retention), macro | 46.3% | 41.5% | 32.2% |

Offline EE absolute-target MAE: Action-only 0.039 / 0.175 / 0.221 cm and Joint-to-Action 0.048 / 0.144 / 0.143 cm
(IID / compositional / visual OOD). In the QA-only isolated-zero control, IID accuracy drops to 6.5% (micro).

Differences from the snapshot recipe, all recorded in the JSON:
- The dataset artifacts were regenerated from the two raw shards with `dataset prepare`. Episode membership matches
  `metadata/episodes.csv` (678 references, 209 execution scenarios), but the generated primary QA has 4,449 items
  (train 2,776, val 415, IID 416, compositional 419, visual OOD 423) instead of 4,469, so `data.strict_snapshot` is off.
- Micro batches QA 2 -> 32 and EE 4 -> 64 (validation / evaluation 4 -> 32) with the global batches unchanged (128 / 256).
- `omnibench/cli.py` sets `allow_abbrev=False`; without it the worker flag `--mode` was parsed as `--model` and every
  training stage failed to start.
- Python 3.11 and PyTorch 2.8 (Transformers 5.3.0, PEFT 0.18.1) instead of the reference PyTorch 2.7.1.

Wall clock including evaluation: QA-only 1 h 37 min, Action-only 3 h 44 min, Joint-to-Action 5 h 03 min. As with the
other profiles, these numbers come from one seed and a different optimization history, so they are not a controlled
model-size comparison.

## Training provenance

The phase-step number below counts updates in the named phase; it is not the complete optimization history. All showcase runs use seed 42. Prior task-specific warm starts are recorded as training provenance, without including exploratory artifacts.

| Model / result | Initialization and selected phase |
|---|---|
| Qwen3.5-2B execution | Task warm start (1,392 joint updates), 2,000 aligned joint updates, then selected action-specialization update 5,500 |
| Llama-3.2-3B execution | Task warm start (1,392 action updates), then 4,000 aligned action updates; fixed evaluated snapshot |
| Qwen3.5-4B Action-only execution | Shared fresh trainable initialization; action-only selected at update 5,800 |
| Qwen3.5-4B Joint-to-Action execution | Shared fresh initialization; selected joint update 800, then selected action-specialization update 4,500 (global batch 512) |
| Qwen3.5-2B QA | Task warm start (1,392 QA updates), then 2,000 aligned QA updates; fixed evaluated snapshot |
| Llama-3.2-3B QA | Task warm start (1,392 QA updates), then 4,000 aligned QA updates; fixed evaluated snapshot |
| Qwen3.5-4B QA | Shared fresh trainable initialization; QA-only selected at update 800 |

The supplied YAML defaults are the portable prospective recipe. Exact historical warm-start weights are not bundled, and the default fresh-start pipeline does not claim to reproduce historical checkpoint hashes. The completed 4B recipe is provided in `configs/qwen35_4b_complete.yaml`; teammate configs retain the default global EE batch of 256. All reported tracks are complete in this snapshot.

The source measurements were imported from completed evaluation artifacts and sanitized to remove server paths. The delivery CSV hash identifies a portable episode-membership export; it is not the byte hash of the original internal JSON manifest. See `metadata/provenance.json` for source code provenance and `results/selected_results.json` for measurement details.
