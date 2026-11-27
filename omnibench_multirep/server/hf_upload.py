"""Stage the selected weights for Hugging Face (host paths replaced by <ROOT>) and upload them as one private repo.
Run on the server (the token is read from stdin, never printed):  python hf_upload.py <ROOT> <repo name> [--dry] < token"""
import glob, json, os, shutil, sys
import torch

ROOT, NAME = sys.argv[1], sys.argv[2]; DRY = "--dry" in sys.argv
STAGE = os.path.join(ROOT, "hf_upload")
RES = os.path.join(ROOT, "results")
TRACKS = {"qa_only": "onedata_server/qwen25_05b/qa/seed42", "action_only": "onedata_server/qwen25_05b/action/seed42",
          "joint_to_action": "onedata_server_staged/qwen25_05b/action/seed42"}   # Joint-to-Action = staged action specialization

def scrub(o):
    if isinstance(o, str): return o.replace(ROOT, "<ROOT>")
    if isinstance(o, dict): return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return type(o)(scrub(v) for v in o)
    return o

def scrub_json(src, dst):
    if os.path.exists(src): json.dump(scrub(json.load(open(src))), open(dst, "w"), indent=1)

shutil.rmtree(STAGE, ignore_errors=True)
# 1. multirep adaptors: 21 settings (2M trainable parameters each, frozen Qwen2.5-0.5B-Instruct)
m = os.path.join(STAGE, "multirep"); os.makedirs(os.path.join(m, "checkpoints")); os.makedirs(os.path.join(m, "results"))
for p in glob.glob(os.path.join(RES, "multirep_aligned", "ckpt_robot", "*", "*.pt")): shutil.copy(p, os.path.join(m, "checkpoints"))
for p in glob.glob(os.path.join(RES, "multirep_aligned", "results_omni_v1", "*", "*.json")): shutil.copy(p, os.path.join(m, "results"))
shutil.copy(os.path.join(RES, "multirep_aligned", "stages_robot.json"), os.path.join(m, "results", "router_stages.json"))
# 2. package: validation-selected trainable weights of the three tracks (LoRA + encoders + projectors + heads)
for name, rel in TRACKS.items():
    src = os.path.join(RES, "package", "checkpoints", rel); dst = os.path.join(STAGE, "package", name); os.makedirs(dst)
    ck = torch.load(os.path.join(src, "best", "final.pt"), map_location="cpu", weights_only=False)
    ck = {k: (v if k == "model" else scrub(v)) for k, v in ck.items()}
    torch.save(ck, os.path.join(dst, "final.pt"))
    for f in ("selection.json", "config.json", "completion.json"): scrub_json(os.path.join(src, f), os.path.join(dst, f))
    scrub_json(os.path.join(src, "best", "final_weights.json"), os.path.join(dst, "final_weights.json"))
if os.path.isdir(os.path.join(RES, "package", "checkpoints", "onedata_server_staged", "qwen25_05b", "joint", "seed42")):
    scrub_json(os.path.join(RES, "package", "checkpoints", "onedata_server_staged", "qwen25_05b", "joint", "seed42", "selection.json"),
               os.path.join(STAGE, "package", "joint_to_action", "joint_stage_selection.json"))
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_model_card.md"), os.path.join(STAGE, "README.md"))

left = [p for p in glob.glob(os.path.join(STAGE, "**", "*.json"), recursive=True) if ROOT in open(p).read()]
assert not left, f"host paths left: {left}"
size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(STAGE, "**", "*"), recursive=True) if os.path.isfile(p))
print(f"staged {STAGE}: {size / 1e6:.0f} MB")
if DRY: sys.exit(0)

from huggingface_hub import HfApi
token = sys.stdin.readline().strip(); api = HfApi(token=token)
user = api.whoami()["name"]; repo = f"{user}/{NAME}"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=STAGE, repo_id=repo, commit_message="OmniBench Qwen2.5-0.5B: multirep adaptors + package tracks")
print("uploaded to https://huggingface.co/" + repo, "(private)")
