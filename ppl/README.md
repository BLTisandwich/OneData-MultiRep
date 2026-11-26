# OneData-OmniBench

Compact language models predict robot end-effector targets and answer grounded questions through a shared external sensory-encoder interface.

This repository provides three experiment tracks: **Action-only**, **QA-only**, and **Joint-to-Action** (joint representation learning followed by action specialization). The language backbone is adapted with LoRA. Qwen models use the same external encoder interface as Llama; native Qwen image features are not used.

## Selected measured results

| Backbone | Execution track | Assisted execution success | QA-only IID macro accuracy |
|---|---|---:|---:|
| Qwen3.5-2B | Joint-to-Action | 130/209 (62.2%) | 67.7% |
| Llama-3.2-3B-Instruct | Action-only | 112/209 (53.6%) | 71.2% |
| Qwen3.5-4B | Joint-to-Action | 132/209 (63.2%) | 66.6% |

Execution and QA columns use **separate checkpoints**. These are selected completed configurations with different training histories, not a matched-budget ranking. All held-out split results for these configurations are retained in [Results](docs/RESULTS.md) and [aggregate JSON](results/selected_results.json). Qwen3.5-4B Joint-to-Action is complete: 132/209 execution successes. Its same-checkpoint QA retention is reported separately in Results; the QA-only column above uses a different checkpoint.

Execution means actual model-driven, assisted simulation with task-goal and release checks. It does not measure physical manipulation. Penetration is recorded as a diagnostic and does not itself cause failure.

## Installation

Use Python 3.10 or 3.11 and install the appropriate PyTorch runtime for your accelerator first. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[simulation,test]'
python -m omnibench --help
python -m omnibench dataset inspect
```

The reference environment uses PyTorch 2.7.1, Transformers 5.3.0 and PEFT 0.18.1. On Ascend, install the matching vendor runtime and torch-npu separately, then activate its environment before running commands. CUDA and Ascend paths are detected at runtime. CPU supports validation and small fixtures; full training requires an accelerator.

## Start a teammate model

The package contains no dataset, encoder checkpoints or LLM weights. Place the shared artifacts according to [Dataset](docs/DATASET.md), or configure their locations in YAML. Every relative path resolves from `project_root`, relative to the YAML file.

```bash
python -m omnibench --config configs/teammate_qwen25_05b.yaml fetch --encoders
python -m omnibench --config configs/teammate_qwen25_05b.yaml dataset check
python -m omnibench --config configs/teammate_qwen25_05b.yaml doctor
python -m omnibench --config configs/teammate_qwen25_05b.yaml run --devices 1 --tracks action qa joint_action
```

For Llama-3.2-1B, use `configs/teammate_llama32_1b.yaml`. Obtain access to the model on Hugging Face and authenticate locally before fetching gated weights. Credentials are never included in configs.

`--devices` is the number of visible accelerators used by each stage. The tracks run sequentially and each stage uses all requested devices. Global batch sizes remain fixed as the device count changes. See [Teammate quickstart](docs/TEAMMATE_QUICKSTART.md) for a small-memory setup and [Training](docs/TRAINING.md) for checkpoint selection and continuation.

## Documentation

- [Data inventory, partitions and exclusions](docs/DATASET.md)
- [Training recipes and saved checkpoints](docs/TRAINING.md)
- [Evaluation and modality evidence](docs/EVALUATION.md)
- [Adding a backbone](docs/ADDING_MODELS.md)
- [Selected results and training provenance](docs/RESULTS.md)
- [Source attribution](NOTICE.md)

Run local checks with `OMNIBENCH_DEVICE=cpu python -m pytest --basetemp=.test_tmp`. Tests create tiny local Qwen2/Llama fixtures, exercise encoder-to-LLM gradients and checkpoint round trips, and validate the frozen partition. No pretrained-model download is required for these checks.

Source and third-party licensing information is in NOTICE.md. Model weights and dataset access are separate from this source delivery.
