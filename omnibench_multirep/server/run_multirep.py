"""Run multirep_robot.ipynb headless (server, no Colab): executes its code cells in order with a few CFG overrides.

The install cell is skipped (the container + pylib provide the packages). Outputs (results_omni_v1/, ckpt_robot/,
results_robot.json, stages_robot.json, figures, run_artifacts_robot.zip) are written to --workdir.
--limit N keeps N train examples (N/4 for the evaluation splits): a quick smoke test of every code path."""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--workdir", required=True)
ap.add_argument("--data", default="omni_v1"); ap.add_argument("--run-set", default="full"); ap.add_argument("--seeds", default="0")
ap.add_argument("--route-setting", default=None); ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

nb = json.load(open(a.notebook, encoding="utf8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert code[0].startswith("# ---------- Install"), "first code cell should be the install cell"
code = code[1:]
cfg = code[0]
def sub(old, new):
    global cfg
    assert cfg.count(old) == 1, f"config line not found: {old}"; cfg = cfg.replace(old, new)
sub('data="robot_v1",', f'data={a.data!r},')
sub('omni_cache="/content/drive/MyDrive/OneData_OmniBench_v1/cache/omni_v1_cache.pkl",', f'omni_cache={os.path.abspath(a.cache)!r},')
sub('run_set="full",', f'run_set={a.run_set!r},')
sub('train_seeds=[0],', f'train_seeds={[int(s) for s in a.seeds.split(",")]},')
sub('route_setting=None,', f'route_setting={a.route_setting!r},')
if a.epochs: sub("epochs=15,", f"epochs={a.epochs},")
code[0] = cfg

os.makedirs(a.workdir, exist_ok=True); os.chdir(a.workdir)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.show = lambda *x, **k: plt.close("all")
g = {"__name__": "__main__", "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x, flush=True)}
t_all = time.time()
for i, c in enumerate(code):
    title = c.strip().splitlines()[0][:100]
    print(f"\n######## cell {i}: {title}", flush=True); t0 = time.time()
    exec(compile(c, f"cell{i}", "exec"), g)
    if a.limit and c.lstrip().startswith("# ---------- Randomized pick-and-place"):   # the data cell only
        n = a.limit
        g["train_data"] = g["train_data"][:n]; g["test_data"] = g["test_data"][: max(8, n // 4)]
        if "router_data" in g: g["router_data"] = g["router_data"][: max(8, n // 4)]
        g["EXTRA_TESTS"] = {k: v[: max(8, n // 4)] for k, v in g["EXTRA_TESTS"].items()}
        print(f"SMOKE: {len(g['train_data'])} train / {len(g['test_data'])} test examples", flush=True)
    print(f"######## cell {i} done in {time.time() - t0:.0f}s", flush=True)
print(f"\nALL CELLS DONE in {(time.time() - t_all) / 60:.1f} min; outputs in {os.getcwd()}", flush=True)
