# Delivery validation

The portable source package was checked on 2026-11-26 using Python 3.10 and CPU PyTorch 2.7.1. The final update includes completed Qwen3.5-4B Joint-to-Action measurements. The latest CPU suite passes 11 tests, with two intentional skips (duplicate architecture trainer coverage and the optional external-data integration test). The real simulator integration check passed separately during package preparation.

- Every Python module compiles and imports.
- CLI help, dataset inspection and environment inspection execute successfully.
- The frozen manifest has 800 unique source IDs, 678 accepted references and the declared train/validation/test memberships. T7 is train-only.
- The package checks the existing 678 shared caches and 4,469 primary QA rows against its frozen membership/counts.
- Tiny, locally initialized Qwen2 and Llama backbones pass multimodal action/QA forward and backward checks, including nonzero finite encoder, projector, LoRA and action-head gradients.
- Trainable-checkpoint save/load preserves outputs for both architecture fixtures.
- Actual optimizer/validation runs cover action, QA and the internal joint stage on tiny fixtures, retaining latest and best checkpoints.
- Causal tactile timing, numerical/discrete QA scoring, release-required goal scoring, Wilson interval boundaries and sampler resume position pass checks.
- A real 96-action T1 expert replay matches the existing aligned reference EE/object trajectories within 0.000001 m using PyBullet 3.2.7.
- The stage-specific batch override is exercised by actual tiny-model optimizer runs and launcher argument-routing checks. Final aggregate counts and checkpoint identity agree across the result files.
- The release audit excludes runtime data/weights/logs, checks text for host-specific paths and credential patterns, and verifies English-only delivery content.

These checks establish package integrity and the tested interfaces. Full-sized training of the teammate backbones, CUDA execution, the new GitHub Actions workflow and fresh end-to-end regeneration of all 678 episodes have not been run as part of this packaging task. The historical three-model results are existing measured runs, not newly reproduced portable-package results.

Run unit checks with:

```bash
OMNIBENCH_DEVICE=cpu python -m pytest --basetemp=.test_tmp
python scripts/check_release.py
```

The optional simulator test runs when `OMNIBENCH_INTEGRATION_EPISODE` and `OMNIBENCH_INTEGRATION_REFERENCE` point to one matching raw/aligned episode pair. It requires simulation dependencies and writes only into the pytest temporary directory.

Build the source archive from the repository root with:

```bash
python scripts/build_release.py
```

The archive contains a per-file SHA-256 manifest and is accompanied by an archive SHA-256 sidecar. No observations or weights are needed to build it.
