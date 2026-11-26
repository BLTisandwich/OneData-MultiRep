# Dataset and frozen partition

## Source and tasks

The original collection has 800 episodes, generated as 100 episodes for each of eight task families. The owners provide their own raw archives; this repository contains IDs and selection flags only.

| Task | Family | Owner | Released use |
|---|---|---|---|
| T1 | Pick & Place | Lee | Training, validation and testing |
| T2 | Push & Slide | Lazizbek | Training, validation and testing |
| T3 | Stack Objects | Lee | Training, validation and testing |
| T4 | Place in Container | Lazizbek | Training, validation and testing |
| T5 | Press Button | Lee | Training, validation and testing |
| T6 | Tap & Knock | Lazizbek | Training, validation and testing |
| T7 | Select & Sort | Lazizbek | Training only; 34 accepted training episodes |
| T8 | Drag & Pull | Lee | Training, validation and testing |

T7 is excluded from every validation and test cohort. The retained T7 training episodes must not be relabeled as held-out examples.

## Partitions

Source episode indices are assigned before generation: 0000–0059 train, 0060–0069 validation, 0070–0079 IID test, 0080–0089 compositional test, and 0090–0099 visual OOD test. An episode and all its views, frames and questions belong to one split.

| Partition | Aligned reference episodes | Primary QA items | Execution scenarios |
|---|---:|---:|---:|
| Train | 418 | 2,789 | — |
| Validation | 66 | 417 | 70 |
| Test IID | 61 | 418 | 69 |
| Test compositional | 67 | 420 | 70 |
| Test visual OOD | 66 | 425 | 70 |
| Total | 678 | 4,469 | 279 including validation |

The offline test reference contains **194 episodes**; the execution test contains **209 scenarios**. Fifteen execution scenarios lack an accepted aligned offline reference but remain execution tests. The original IID scenario `T1_0074` is excluded from execution. The precise IDs and decisions are in `metadata/episodes.csv`; do not infer membership from directory contents.

Each row has `train_reference`, `validation_reference`, `validation_execution`, `test_reference`, and `test_execution` flags. `scope_note` explains its cohort status. The CSV includes all 800 source IDs, including excluded rows. `metadata/snapshot.json` stores the accepted counts and CSV hash.

Compositional testing holds out a generator goal-side/location pattern. Visual OOD changes the held-out object/background palette and associated spoken color content; it is not an unseen-mesh benchmark or a pure background-only change.

## Sensor interface

Episodes have 96 synchronized observations over approximately eight seconds (12 FPS) and a 240 Hz simulation trace. Cameras are front, overhead and side at 192 × 144. There are 95 valid offline action examples per accepted reference episode: 39,710 training frames and 18,430 held-out reference frames.

| Encoder branch | Input and provenance |
|---|---|
| vision | Front RGB; frozen ResNet18 image features |
| multiview | Overhead and side RGB features from the same ResNet18 |
| depth | Camera-aligned depth |
| segmentation | Instance masks |
| flow | Estimated optical flow |
| points | Depth-unprojected points in world coordinates |
| mesh | Procedural target mesh vertices and adjacency |
| speech | Synthesized instruction WAV; frozen Whisper-tiny features |
| impact | Synthesized event/impact audio as log-mel windows |
| timeseries | EE motion, joints, velocities and finger state |
| force | Simulation contact signals; zero can be a valid observation |
| tactile | Heuristic dual-finger pressure proxy |
| proprio | EE pose, joints, joint velocities and fingers |

Synthetic audio and tactile proxies are accepted inputs, not recordings from physical sensors. Assisted attachments and virtual press/tap events can produce valid zero physical force. Do not replace zeros with invented physical measurements. The action view reconstructs causal tactile signals so future virtual events cannot appear in earlier inputs.

## Layout for existing shared artifacts

```text
data/
  raw/
    Lee/episodes/T1_0000/...
    Lazizbek/episodes/T2_0000/...
  aligned/episodes/T1_0000/...
  cache/T1_0000.pt
  static/T1_0070.pt
  qa/
    episodes.jsonl
    qa_primary.jsonl
    feature_controls_grade_only.jsonl
    checksums.json
assets/
  models/<model-directory>/...
  encoders/resnet18.pt
  encoders/whisper-tiny/...
```

If aligned episodes span multiple folders, list them under `aligned_roots` in YAML. The cache may also be one directory containing symlinks to multiple shared cache folders. Configure `paths.cache`, `paths.qa`, `paths.static_cache` and `raw_roots` accordingly. The loader rebases old cached `source_path` metadata to the configured aligned episode; no old server path is used.

Training needs the aligned references, the 678 per-episode caches and QA manifests. Execution additionally needs the original raw episodes, frozen visual encoder and 279 static instruction/mesh caches. QA gold and intervention answers are grading artifacts, not model inputs. Raw `task.json` is used by the simulator and grader; its oracle goal fields are not passed into the model.

```bash
python -m omnibench --config configs/default.yaml dataset inspect
python -m omnibench --config configs/default.yaml dataset check
```

`inspect` verifies the partition metadata without loading observations. `check` additionally checks cache/QA presence, episode membership and split/count consistency; it does not imply a full checksum audit of every raw sensor file. Per-episode `_COMPLETE.json` files retain sensor checksums.

## Rebuild when only raw episodes are available

Fetch the shared encoders, set both raw owner directories in YAML, and use:

```bash
python -m omnibench fetch --encoders
python -m omnibench dataset prepare --stage align --workers 4
python -m omnibench dataset prepare --stage cache
python -m omnibench dataset prepare --stage static
python -m omnibench dataset prepare --stage qa
python -m omnibench dataset check
```

Alignment replays the source expert targets under the executed-state simulator and retains the original commanded EE supervision. It writes a separate derived directory. It enforces the frozen accepted reference IDs; a simulator-version mismatch raises an error instead of silently changing the cohort. Cache preparation uses the shared pretrained encoders. QA generation is deterministic and uses saved evidence without paid API calls.

Alignment and QA preparation run once with CPU workers. Cache/static preparation can be sharded with `torchrun --standalone --nproc-per-node=4 -m omnibench dataset prepare --stage cache` (and `--stage static`). Run these distributed stages separately; `--stage all` is intended for a single process.
