---
license: other
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- robotics
- multimodal
- question-answering
- onedata-omnibench
---

# OneData-OmniBench: Qwen2.5-0.5B weights

Trained weights from the KAIST AIC.30300 project *OneData-MultiRep*, on the OneData-OmniBench episodes (simulated Franka
Panda tabletop manipulation, 8 tasks, 678 aligned reference episodes). The base LLM is **not** included; load
`Qwen/Qwen2.5-0.5B-Instruct` from the Hub. Code: the project repository (`multirep_robot.ipynb`, `omnibench_cache.ipynb`,
`omnibench_multirep/`, `ppl/`).

## Contents

### `multirep/`: budget-matched multi-representation adaptors (frozen LLM)

`checkpoints/*.pt`: 21 adaptors, **2M trainable parameters each** (encoders + fusion + aggregator), feeding 16 soft
tokens to the frozen Qwen2.5-0.5B-Instruct: blind, single representations (view, point, proprio, speech, tactile, impact)
and fusion designs. Each file holds `state_dict` and the constructor arguments (`reps`, `agg`, `width`, `K`, `llm_dim`,
`q_cond`, `out_rms`, `fusion`, ...); rebuild with `MultiRepAdaptor` from `multirep_robot.ipynb` (section 7b, `load_run`).
`results/*.json`: per-setting accuracies on the six frame questions (test IID, compositional, visual OOD);
`results/router_stages.json`: router inputs / outputs.

Test IID accuracy: fusion settings .96-.99, best single representation .85-.90, blind .30. Compositional: fusion
.78-.85 vs. <= .65; visual OOD: fusion .62-.76 vs. <= .61. Router at lambda*: .987 with 27 % of encoder parameters.

### `package/`: the three tracks of the `ppl/` package (LoRA)

Validation-selected trainable weights (LoRA rank 16 on q/v, sensor encoders, projectors, heads) of
`qa_only`, `action_only` and `joint_to_action` (action specialization after the joint stage; the joint-stage selection is
in `joint_to_action/joint_stage_selection.json`). Load with `ppl/omnibench` (`evaluation.load_model`) together with the
base model and the shared frozen encoders (ResNet18 ImageNet weights, openai/whisper-tiny).

| track | assisted execution success (209) | IID | compositional | visual OOD |
|---|---|---|---|---|
| Action-only | 116 (55.5 %) | 67/69 | 38/70 | 11/70 |
| Joint-to-Action | 126 (60.3 %) | 61/69 | 39/70 | 26/70 |

QA-only, encoder-category macro accuracy (full observation): IID .631, compositional .552, visual OOD .440.

## Training setup

One NVIDIA A100-SXM4-40GB per job (GIST cluster), seed 0 (adaptors) / 42 (package). Adaptors: 15 epochs, batch 64,
AdamW 5e-4, fp16 autocast. Package: teammate recipe (`ppl/configs/teammate_qwen25_05b_a100.yaml`), bf16; micro batches
raised with the same global batches; QA regenerated from the raw shards (4,449 primary items vs. 4,469 in the frozen
snapshot). One training seed throughout.

## Limitations

Simulated, assisted kinematic manipulation (scripted expert, assisted grasps, synthetic audio / tactile proxies); not a
physical or real-robot result. Frame questions are templated and near ceiling on IID. Single seed.
