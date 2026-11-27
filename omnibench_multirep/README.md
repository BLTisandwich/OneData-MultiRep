# OmniBench frame QA with a budget-matched multi-representation adaptor (Qwen2.5-0.5B)

The project question on the team benchmark: **under a fixed trainable-parameter budget, does one representation-specific
encoder suffice, or do several small encoders fused into the soft tokens of a frozen LLM do better?** And can a router
pick, per question, only the representations it needs?

This folder holds the code that ran it on the OmniBench episodes, the measured results and the figures. The adaptor
itself is `multirep_robot.ipynb` (repository root, `data="omni_v1"`); the cache builder is `omnibench_cache.ipynb`.

| | this folder (`multirep_robot.ipynb`) | `ppl/` package |
|---|---|---|
| LLM | Qwen2.5-0.5B-Instruct, **frozen** | LoRA rank 16 |
| inputs | view (3 cameras), point cloud, proprio, speech (+ tactile, impact proxies) | 13 branches |
| trainable budget | **2M per setting, matched** (encoders + fusion + aggregator) | not matched |
| task | frame-level grounded QA (6 questions) | episode QA, EE action, assisted execution |
| episodes / splits | **the same 678 aligned episodes and splits** (train 418, val 66, IID 61, compositional 67, visual OOD 66) | |

The question sets differ, so accuracies are not comparable across the two columns.

## Pipeline

1. **Episodes.** The 800 generated episodes (Lee + Lazizbek shards) are aligned by the package
   (`python -m omnibench dataset prepare --stage align`), which keeps the frozen 678 reference episodes. Aligned episodes
   do not store the expert phase; the cache takes it from the raw episode (same waypoint schedule).
2. **Cache** (`omnibench_cache.ipynb`, CPU). Per sampled frame: 3 camera views (128 px), a colored point cloud rebuilt
   from depth (table removed, 768 points), proprio (12 frames x 11), the spoken instruction (log-mel), and causal windows
   of the tactile and impact-audio proxies. Six frame questions are sampled balanced over answers (1,200 per question in
   train, 150 in every other split), and their answers come from the saved simulator state.
3. **Sweep** (`multirep_robot.ipynb`, `data="omni_v1"`). 21 settings at 2M trainable parameters each, 15 epochs:
   blind, single representations, fusion designs (Q-Former, per-representation queries, MLP, relational fusion, joint
   attention, view encoder v2, asymmetric budget), plain training, and 6-representation models with the proxies.
4. **Router** (same notebook, section 9). It reads the question only and picks a subset of the best fusion model's
   encoders; trained on val, evaluated on test.
5. **Real-time QA demo** (`server/omni_demo.py`). The routed fusion model and the single-representation and blind models
   answer the six questions at every frame of held-out episodes by free greedy generation, with mp4 output.

| question | answer | needs |
|---|---|---|
| task | pick place / stack / ... (8) | speech |
| color | blue / orange / green (magenta in visual OOD) | speech, view, point |
| motion | up / down / left / right / forward / backward | proprio |
| next_action | expert phase 0.5 s ahead | proprio + speech |
| goal_side | closer / farther (goal vs object from the robot base) | point, view |
| goal_dist | goal within 33 cm of the object? | point, view |

## Results (one seed; full tables in [results/tables.md](results/tables.md))

- **Fusion beats every single representation at the same budget**: test IID .96–.99 for all fusion settings vs. point
  only .85, view only (v2) .90, proprio only .72, speech only .61, blind .30. The gap widens on the held-out splits
  (compositional .78–.85 vs. ≤ .65; visual OOD .62–.76 vs. ≤ .61).
- **Each representation answers what it should**: proprio solves motion / next_action, point solves color / goal_side /
  goal_dist, speech solves task / color; blind stays at chance.
- **Router**: at lambda* the routed model keeps test accuracy .987 (all encoders: .988) while running 27 % of the encoder
  parameters (1.5 representations per question), and its choices follow the task design (task, color -> speech;
  goal questions -> point; motion -> point + proprio; next_action -> point + proprio + speech).
- **Demo** (test IID): over seven held-out episodes the routed fusion model answers .93–.98 of the graded frames; the
  best single representation (point) .63–.87; blind and view (v1) collapse. Compositional demo: fusion .81–.96, best
  in every episode. Visual OOD demo: fusion .62–.72; proprio alone (color-independent) reaches .29–.68, above fusion on
  one episode (T2) and within .04 of it on two (T5, T8). Per-episode tables in [results/tables.md](results/tables.md).
- IID accuracy is near ceiling, so the fusion designs differ mostly on the held-out splits.

Figures: [router frontier](figures/router_frontier.png), [subset robustness](figures/router_subset_robustness.png),
[model inputs](figures/input_examples.png), [demo accuracy by model](figures/demo_accuracy_by_model.png),
demo key frames ([T1](figures/demo_T1_0070_keyframes.png), [T4](figures/demo_T4_0070_keyframes.png),
[T5](figures/demo_T5_0070_keyframes.png)).

**Demo videos** (8 s each, 1280x720, with the spoken instruction as audio): `videos/<split>/demo_<episode>.mp4`, one
held-out episode per task for each test split (`test_iid`, `test_compositional`, `test_visual_ood`). Each frame shows
the routed fusion model's answer to the six questions and, below it, the answers of the blind and single-representation
models of the same sweep.

**Weights**: the 21 adaptors (and the `ppl/` Qwen2.5-0.5B track weights) are on the Hugging Face Hub, repository
`lhs0520/onedata-omnibench-qwen05b` (private until submission).

## Compute and reproducibility

| item | value |
|---|---|
| hardware | GIST cluster node `hpc-pr-a-pod12`, 1 x NVIDIA A100-SXM4-40GB per job (Slurm), 8–32 CPU cores |
| software | Singularity container: Python 3.11.13, PyTorch 2.8.0+cu128, Transformers 4.57.3; pybullet 3.2.7, imageio-ffmpeg 0.8.1 |
| LLM | Qwen/Qwen2.5-0.5B-Instruct from Hugging Face, frozen fp32 weights, fp16 autocast |
| adaptor training | 15 epochs, batch 64, AdamW lr 5e-4, wd 0.01, grad clip 1.0, balance recipe (modality dropout, auxiliary heads, OGM damping, per-encoder lr); seed 0 |
| wall clock | cache 1.4 min (CPU, 32 cores); sweep 106 min for 21 settings (~5 min each) + router; demo 4 min (7 episodes x 6 models); Lee-shard pilot 26 min |
| external services | Google Drive (episode transfer), Hugging Face Hub (model download); no paid API |

Reproduce on the same cluster (paths in `server/env.sh`):

```bash
sbatch server/jobs/prepare.sbatch                     # package: align / cache / static / QA (aligned episodes)
sbatch server/jobs/omnicache.sbatch                   # our frame cache from the aligned episodes
sbatch server/jobs/multirep.sbatch <cache.pkl> aligned --run-set full
sbatch --gres=gpu:1 server/jobs/demo.sbatch <sweep dir> <cache.pkl> <aligned dir> --run-set full --per-task 1
python omnibench_multirep/make_tables.py              # results/tables.md from the JSON files
```

On Colab: build the cache with `omnibench_cache.ipynb`, then run `multirep_robot.ipynb` with `data="omni_v1"` and
`omni_cache` pointing to it.

## Limitations

- One training seed; differences of a few points between fusion designs are within noise.
- Frame questions are templated and balanced; IID accuracy saturates. The held-out splits carry the comparison.
- Visual OOD `color` answers (magenta) never occur in training, so that cell is 0 for every model by construction.
- The tactile and impact-audio signals are simulator proxies derived from the grasp state and event log.
- Episode content: kinematic scripted expert with assisted grasps; no physical force closure.

## Files

- `results/runs/*.json`: one file per setting (accuracies per question and split, parameter counts, hardware, timing).
- `results/lee_pilot/*.json`: pilot on the Lee shard only (raw episodes, 4 tasks, 5 key settings).
- `results/router_stages.json`: router inputs and outputs (subset scores, lambda sweep, final table).
- `results/demo_summary*.json`: demo accuracy per episode, question and model (one file per test split).
- `results/router_choices.md`: which subset the router picks per question.
- `videos/<split>/`: real-time QA demo videos, one held-out episode per task.
- `results/cache_summary.json`: cache statistics and shortcut baselines of the frame questions.
- `server/`: Slurm / Singularity jobs and the headless runners used on the cluster (`omni_demo.py`: demo videos;
  `hf_upload.py` + `hf_model_card.md`: weights upload); `server/drive_to_tar.ipynb` packs the Drive shards into one tar
  per owner.
- `make_tables.py`: builds `results/tables.md`.
