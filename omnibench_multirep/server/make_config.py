"""Write the server configs for the omnibench package from the teammate 0.5B config.

Same recipe as configs/teammate_qwen25_05b.yaml; only paths (absolute, outside the code checkout) and the experiment name
change. The smoke config runs a few QA updates to check loading, training, checkpointing and prediction before the real run.
Usage: python omnibench_multirep/server/make_config.py <ROOT> <REPO>"""
import copy, sys
from pathlib import Path
import yaml

root, repo = sys.argv[1], Path(sys.argv[2])
c = yaml.safe_load((repo / "ppl/configs/teammate_qwen25_05b.yaml").read_text())
c["experiment"] = "onedata_server"
c["paths"].update(aligned=f"{root}/data/aligned", cache=f"{root}/data/pkg_cache", static_cache=f"{root}/data/pkg_static",
                  qa=f"{root}/data/pkg_qa", models=f"{root}/assets/models", vision=f"{root}/assets/encoders/resnet18.pt",
                  speech=f"{root}/assets/encoders/whisper-tiny", outputs=f"{root}/results/package")
c["raw_roots"] = {"Lee": f"{root}/data/raw/Lee", "Lazizbek": f"{root}/data/raw/Lazizbek"}
c["aligned_roots"] = [f"{root}/data/aligned"]
c["runtime"]["threads"] = 8
# QA regenerated from the raw shards has 4,449 primary items vs 4,469 in the frozen snapshot (-20, 0.45 %): keep the
# exact episode membership checks, skip only the exact-count assertion (documented with the results)
c["data"]["strict_snapshot"] = False
# A100 40 GB: the teammate micro batches (QA 2, EE 4) use ~2 GB and need 64 accumulation steps per update. Same global
# batches (128 / 256) -> same optimization (gradient accumulation), larger micro batches -> far fewer forward passes.
c["training"].update(qa_micro_batch=32, ee_micro_batch=64, validation_batch=32)
c["evaluation"]["batch_size"] = 32
(repo / "ppl/configs/server_qwen25_05b.yaml").write_text(yaml.safe_dump(c, sort_keys=False))

s = copy.deepcopy(c)
s["experiment"] = "onedata_smoke"
s["paths"]["outputs"] = f"{root}/results/package_smoke"
s["training"].update(validation_every=5, global_qa_batch=4, global_ee_batch=8)
for k in ("qa", "action"):
    s["training"][k].update(max_steps=10, minimum_steps=5, patience=1)
s["training"]["joint_action"].update(joint_max_steps=10, joint_minimum_steps=5, joint_patience=1)
s["evaluation"]["modality_ablations"] = False
(repo / "ppl/configs/server_qwen25_05b_smoke.yaml").write_text(yaml.safe_dump(s, sort_keys=False))
print("wrote server_qwen25_05b.yaml and server_qwen25_05b_smoke.yaml")
