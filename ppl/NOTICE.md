# Source and third-party notices

The generator-derived scene, waypoint, canonical-QA and validation helpers originate from BLTisandwich/OneData-MultiRep, pinned at commit c0494a6076b024d9964d358ac39356483470a029. Source notebooks are identified in metadata/provenance.json.

Source: https://github.com/BLTisandwich/OneData-MultiRep

The portable runtime adapts the source simulator to executed-state assistance and a shared external-encoder training/evaluation interface. The upstream source snapshot does not contain a project license grant; this delivery does not assert an additional license for upstream-derived code. Confirm project licensing with its owners before wider redistribution.

`omnibench/attention.py` contains an adapted PyTorch Qwen3.5 gated-DeltaNet computation from Hugging Face Transformers 5.3. The original project is copyright its authors and is distributed under Apache License 2.0. The full supplied Transformers license is in `third_party/TRANSFORMERS_LICENSE.txt`. The adaptation replaces the triangular recurrence implementation with an algebraically equivalent finite-series computation and adds runtime precision wrappers.

Other runtime dependencies remain under their own licenses. Hugging Face model access and model licenses apply separately. ResNet18 and Whisper encoder weights, language-model weights and dataset files are not included in this delivery.
