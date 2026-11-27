"""Execute the code cells of a notebook headless (used for omnibench_cache.ipynb on the server; config via its env vars)."""
import json, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.show = lambda *a, **k: plt.close("all")
nb = json.load(open(sys.argv[1], encoding="utf8"))
g = {"__name__": "__main__", "display": print}
for i, c in enumerate("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"):
    t0 = time.time(); exec(compile(c, f"cell{i}", "exec"), g); print(f"## cell {i} done in {time.time() - t0:.0f}s", flush=True)
