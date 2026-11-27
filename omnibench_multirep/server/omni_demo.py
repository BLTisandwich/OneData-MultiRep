"""Real-time QA demo on OmniBench episodes: the trained (routed) multirep adaptor answers the six frame-level questions
at every frame of held-out episodes, by free greedy generation from the frozen LLM. One mp4 per episode (+ all of them
concatenated), key-frame figures and a JSON summary.

Inputs are built exactly as in training: the frame functions of omnibench_cache.ipynb are executed from the notebook
itself. The model, its checkpoints and the router come from a finished multirep_robot run (--workdir: the sweep's
working directory, the same CFG overrides as that run). --dry: no model (answers = ground truth), to test the rendering.

python omnibench_multirep/server/omni_demo.py --workdir <sweep workdir> --cache <omni_v1 cache .pkl> --episodes-root <dir with episodes/>
       [--episodes T1_0070,T2_0071 | --per-task 1] [--split test_iid] [--every 1] [--run-set full --epochs N] [--dry]"""
import argparse, ast, json, os, subprocess, sys, time, zlib
import numpy as np

ap = argparse.ArgumentParser()
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))   # repo root
ap.add_argument("--notebook", default=os.path.join(REPO, "multirep_robot.ipynb"))
ap.add_argument("--cache-notebook", default=os.path.join(REPO, "omnibench_cache.ipynb"))
ap.add_argument("--workdir", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--episodes-root", required=True)
ap.add_argument("--episodes", default=""); ap.add_argument("--per-task", type=int, default=1); ap.add_argument("--split", default="test_iid")
ap.add_argument("--every", type=int, default=1); ap.add_argument("--out", default="demo_omni")
ap.add_argument("--run-set", default="full"); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--seeds", default="0")
ap.add_argument("--limit", type=int, default=0)   # same as the run being shown (smoke runs)
ap.add_argument("--compare", default="blind (no inputs),proprio only,point only,view only,speech only",
                help="baseline settings of the same sweep answering the same frames (comma-separated; '' = none)")
ap.add_argument("--route-setting", default=None); ap.add_argument("--no-route", action="store_true"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
a.workdir, a.cache, a.episodes_root = map(os.path.abspath, (a.workdir, a.cache, a.episodes_root))
os.makedirs(a.workdir, exist_ok=True); os.chdir(a.workdir); os.makedirs(a.out, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches
plt.show = lambda *x, **k: plt.close("all")
try:
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    import shutil; FFMPEG = shutil.which("ffmpeg")
if FFMPEG: plt.rcParams["animation.ffmpeg_path"] = FFMPEG
cells = lambda path: ["".join(c["source"]) for c in json.load(open(path, encoding="utf8"))["cells"] if c["cell_type"] == "code"]

# ---- 1. frame functions of omnibench_cache.ipynb (own namespace): setup + config cells, then only the function
#         definitions and constants of the other cells (their pipeline code is not run)
os.environ.update(OMNI_ROOTS=a.episodes_root, OMNI_OUT=os.devnull)
C = {"__name__": "omni_cache"}; cc = cells(a.cache_notebook)
for c in cc[:2]: exec(compile(c, "cache_setup", "exec"), C)
KEEP = {"MEL_FB", "N_FFT", "HOP", "LOG0", "IMPACT_W", "QUESTIONS", "TASKS", "MOTIONS", "AHEAD", "PC_CENTER", "PC_SCALE"}
def defs(node):
    for n in node.body if hasattr(node, "body") else []:
        if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)): yield n
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in KEEP or
                                               isinstance(t, ast.Tuple) and any(getattr(e, "id", "") in KEEP for e in t.elts)
                                               for t in n.targets): yield n
        elif isinstance(n, ast.If): yield from defs(n)
for c in cc[2:]:
    mod = ast.Module(body=list(defs(ast.parse(c))), type_ignores=[]); exec(compile(mod, "cache_defs", "exec"), C)
for f in ("load_light", "frame_inputs", "causal_window", "candidates"): assert f in C, f"{f} not found in {a.cache_notebook}"

# ---- 2. the trained model + router (skipped with --dry)
G = {"__name__": "__main__", "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x, flush=True)}
if not a.dry:
    code = cells(a.notebook)[1:]                                        # without the install cell
    cfg = code[0]
    for old, new in [('data="robot_v1",', 'data="omni_v1",'),
                     ('omni_cache="/content/drive/MyDrive/OneData_OmniBench_v1/cache/omni_v1_cache.pkl",', f"omni_cache={a.cache!r},"),
                     ('run_set="full",', f"run_set={a.run_set!r},"), ("train_seeds=[0],", f"train_seeds={[int(s) for s in a.seeds.split(',')]},"),
                     ("route_setting=None,", f"route_setting={a.route_setting!r},")] + ([("epochs=15,", f"epochs={a.epochs},")] if a.epochs else []):
        assert cfg.count(old) == 1, old; cfg = cfg.replace(old, new)
    code[0] = cfg
    SKIP = ("# ---------- Look at the data", "# ---------- Which representation", "# ---------- 10. Final", "# ---------- 11. Demo")
    for i, c in enumerate(code):
        if c.lstrip().startswith(SKIP): continue
        print(f"## notebook cell {i}: {c.strip().splitlines()[0][:90]}", flush=True); exec(compile(c, f"cell{i}", "exec"), G)
        if a.limit and c.lstrip().startswith("# ---------- Randomized pick-and-place"):   # the data cell, as in run_multirep.py
            n = a.limit; G["train_data"] = G["train_data"][:n]; G["test_data"] = G["test_data"][: max(8, n // 4)]
            G["router_data"] = G["router_data"][: max(8, n // 4)]; G["EXTRA_TESTS"] = {k: v[: max(8, n // 4)] for k, v in G["EXTRA_TESTS"].items()}
    import torch
    model = G["route_model"]; REPS = list(model.reps); QS = G["QUESTIONS"]; TASKS = list(G["TASKS"])
    if a.no_route: use = {t: tuple(REPS) for t in TASKS}
    else:
        rt = G["train_router"](G["X_router"], G["L_router"], G["LAM_STAR"])
        with torch.no_grad(): pick = rt(G["question_feats"]([dict(q=QS[t]) for t in TASKS])).argmax(-1).tolist()
        use = {t: G["SUBSETS"][j] for t, j in zip(TASKS, pick)}
    cost = {t: sum(G["ENC_PARAMS"][r] for r in use[t]) / sum(G["ENC_PARAMS"].values()) for t in TASKS}
    LABEL = f"Frozen {G['BACKBONE']} + adaptor: {G['ROUTE_SETTING']}" + ("" if a.no_route else f", router lambda*={G['LAM_STAR']}")

    # baselines: single-representation and blind models of the same sweep (same budget), all their inputs, no router
    BASE = {}
    for name in [s.strip() for s in a.compare.split(",") if s.strip()]:
        try: BASE[name.replace(" (no inputs)", "").replace(" only", "")] = G["load_run"](name, G["BUDGETS"][0])
        except Exception as e: print(f"baseline {name!r} not available ({type(e).__name__}): skipped", flush=True)
    print("baselines:", list(BASE), flush=True)

    @torch.no_grad()
    def greedy(exs, drop_reps=(), max_new=6, m=None):
        """Free generation; one question per call, so all prompts have the same length (no padding)."""
        m = m or model; m.eval(); b = G["to_dev"](G["collate"](exs)); assert not b["q_pad"].any()
        if m.q_cond: b["q_emb"] = G["embed"](b["q_ids"]).float()
        drop = torch.tensor([r in drop_reps for r in m.reps], dtype=torch.bool, device=G["device"]).expand(len(exs), -1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=G["AMP"]):
            x = torch.cat([m(b, drop).to(G["LLM_DTYPE"]), G["embed"](b["q_ids"])], 1); out = []
            for _ in range(max_new):
                nxt = G["LM_HEAD"](G["DECODER"](inputs_embeds=x).last_hidden_state[:, -1]).argmax(-1)
                out.append(nxt); x = torch.cat([x, G["embed"](nxt)[:, None]], 1)
        EOS = G["EOS"]; out = torch.stack(out, 1).tolist()
        return [G["tok"].decode(o[:o.index(EOS)] if EOS in o else o).strip() for o in out]
else:
    QS, TASKS = C["QUESTIONS"], list(C["QUESTIONS"]); REPS = ["view", "point", "proprio", "speech"]
    use = {t: tuple(REPS) for t in TASKS}; cost = {t: 1.0 for t in TASKS}; LABEL = "DRY RUN (answers = ground truth)"
    BASE = {s: None for s in ("blind", "proprio", "point", "view", "speech")}   # dry: fake answers, layout test only

# ---- 3. episodes: given ids, or --per-task episodes of the split, taken from the cache's episode list
import pickle
cache = pickle.load(open(a.cache, "rb"))
if a.episodes: ids = a.episodes.split(",")
else:
    by = {}
    for e in cache["splits"][a.split]["eps"]: by.setdefault(e["task_id"], []).append(e["id"])
    ids = [i for t in sorted(by) for i in sorted(by[t])[: a.per_task]]
del cache
print("episodes:", ids, flush=True)

REP_COLOR = {"view": "#2a78d6", "point": "#eb6834", "proprio": "#6250d6", "speech": "#e87ba4", "tactile": "#1baf7a", "impact": "#eda100"}
GOOD, BAD, MUTED, INK = "#1baf7a", "#e34948", "#9a9893", "#1f1e1c"
summary, videos = {}, []
for eid in ids:
    t0 = time.time(); d = os.path.join(a.episodes_root, "episodes", eid)
    ep = C["load_light"](d)
    with np.load(os.path.join(d, "multiview_rgb_depth_seg_flow.npz")) as z:
        cams = {f"{c}_{x}": z[f"{c}_{x}"] for c in C["CAMS"] for x in ("rgb", "depth_mm", "segmentation_int32")}
    frames = list(range(0, ep["T"], a.every)); inp = {}
    for k in frames:
        rng = np.random.default_rng([C["CFG"]["seed"], zlib.crc32(ep["id"].encode()), k])
        inp[k] = C["frame_inputs"](ep, k, cams, rng) + C["causal_window"](ep, k)
    truth = {(t, k): str(ans) for t, k, ans, _ in C["candidates"](ep)}
    answers = {}; base_ans = {s: {} for s in BASE}
    for t in TASKS:                                       # every question at every frame; graded where a ground truth exists
        exs = [dict(task=t, q=QS[t], a=truth.get((t, k), "?"), views=inp[k][0], points=inp[k][1], point_rgb=inp[k][2],
                    proprio=inp[k][3], tactile=inp[k][4], impact=inp[k][5], speech=ep["mel"]) for k in frames]
        if a.dry:
            preds = [ex["a"] for ex in exs]
            for j, s_ in enumerate(BASE): base_ans[s_].update({(t, k): (ex["a"] if (k // 12 + j) % 2 else "left") for k, ex in zip(frames, exs)})
        else:
            preds = []
            for s in range(0, len(exs), 64): preds += greedy(exs[s:s + 64], tuple(r for r in REPS if r not in use[t]))
            for s_, m_ in BASE.items():
                bp = []
                for s in range(0, len(exs), 64): bp += greedy(exs[s:s + 64], (), m=m_)
                base_ans[s_].update({(t, k): p for k, p in zip(frames, bp)})
        answers.update({(t, k): p for k, p in zip(frames, preds)})
    def score_of(ans):
        g = {t: [ans[t, k] == truth[t, k] for k in frames if (t, k) in truth] for t in TASKS}
        allg = sum(g.values(), [])
        return {t: (float(np.mean(v)), len(v)) for t, v in g.items() if v}, (float(np.mean(allg)) if allg else None)
    acc, overall = score_of(answers)
    summary[eid] = dict(task=ep["task_name"], split=ep["split"], instruction=ep["instruction"], accuracy=acc, overall=overall,
                        baselines={s: dict(zip(("accuracy", "overall"), score_of(v))) for s, v in base_ans.items()})
    print(f"{eid} ({ep['task_name']}): fusion " + ", ".join(f"{t} {v:.2f} ({n})" for t, (v, n) in acc.items())
          + f" | overall {overall:.2f} | " + ", ".join(f"{s} {summary[eid]['baselines'][s]['overall']:.2f}" for s in BASE)
          + f" [{time.time() - t0:.0f}s]", flush=True)

    # ---- figure: scene + what the encoders get (left), the six questions (right)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    ax_hi = fig.add_axes([0.01, 0.30, 0.42, 0.60]); ax_hi.axis("off"); im_hi = ax_hi.imshow(cams["front_rgb"][frames[0]])
    title = fig.text(0.015, 0.955, "", fontsize=13, weight="bold", color=INK)
    fig.text(0.015, 0.925, f"{eid} ({ep['task_name']}, {ep['split']})   |   spoken instruction: \"{ep['instruction']}\"", fontsize=9, color="#52514e")
    small = [fig.add_axes([0.012 + i * 0.108, 0.035, 0.1, 0.2]) for i in range(4)]
    others = [c for c in C["CAMS"] if c != "front"]
    ims = [small[i].imshow(cams[f"{c}_rgb"][frames[0]]) for i, c in enumerate(others)]
    for i, c in enumerate(others): small[i].set_title(f"{c} camera", fontsize=7.5)
    P0 = inp[frames[0]][1] * C["PC_SCALE"] + C["PC_CENTER"]
    sc = small[2].scatter(P0[:, 0], P0[:, 1], c=inp[frames[0]][2], s=2); small[2].set_title("point cloud (top view)", fontsize=7.5)
    small[2].set_xlim(0.15, 0.75); small[2].set_ylim(-0.35, 0.35); small[2].invert_xaxis()
    small[3].imshow(ep["mel"].T, origin="lower", aspect="auto", cmap="magma"); small[3].set_title("speech (log-mel)", fontsize=7.5)
    for s_ in small: s_.set_xticks([]); s_.set_yticks([])
    ax_pr = fig.add_axes([0.45, 0.035, 0.12, 0.2]); tt = np.arange(ep["T"]) / C["FPS"]
    for i, (lbl, col) in enumerate(zip("xyz", ["#2a78d6", "#eb6834", "#1baf7a"])): ax_pr.plot(tt, ep["ee"][:, i], color=col, lw=1, label=lbl)
    cursor = ax_pr.axvline(0, color=INK, lw=1); ax_pr.set_title("proprio: gripper xyz", fontsize=7.5); ax_pr.legend(fontsize=6); ax_pr.tick_params(labelsize=6)
    ax_q = fig.add_axes([0.6, 0.02, 0.39, 0.90]); ax_q.axis("off"); ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 1)
    ax_q.text(0, 0.995, LABEL + ("   |   below: single-representation / blind models, same budget" if BASE else ""),
              fontsize=6.5, color="#52514e", va="top")
    rows, brows = {}, {}
    for i, t in enumerate(TASKS):
        y = 0.94 - i * 0.157
        ax_q.add_patch(patches.FancyBboxPatch((0, y - 0.13), 1, 0.145, boxstyle="round,pad=0.004,rounding_size=0.01", fc="#f5f4f1", ec="none"))
        ax_q.text(0.02, y, QS[t], fontsize=8.5, color=INK, va="center")
        gap = min(0.15, 0.58 / len(REPS))                          # lights fit left of the answer column
        for j, r in enumerate(REPS):
            on = r in use[t]
            ax_q.add_patch(patches.Circle((0.025 + j * gap, y - 0.06), 0.008, fc=REP_COLOR.get(r, INK) if on else "white", ec=REP_COLOR.get(r, INK), lw=1.1))
            ax_q.text(0.04 + j * gap, y - 0.06, r, fontsize=6 if len(REPS) > 4 else 6.5, va="center", color=INK if on else MUTED)
        rows[t] = (ax_q.text(0.62, y - 0.035, "", fontsize=11.5, weight="bold", va="center"),
                   ax_q.text(0.97, y - 0.035, "", fontsize=14, weight="bold", va="center", ha="right"),
                   ax_q.text(0.62, y - 0.075, "", fontsize=6.5, va="center", color="#52514e"))
        bw = 0.98 / max(len(BASE), 1)
        brows[t] = {s: ax_q.text(0.02 + j * bw, y - 0.108, "", fontsize=6.3, va="center") for j, s in enumerate(BASE)}
    score = fig.text(0.6, 0.955, "", fontsize=7.5, color=INK, va="center", linespacing=1.4)

    def draw(k):
        im_hi.set_data(cams["front_rgb"][k])
        for i, c in enumerate(others): ims[i].set_data(cams[f"{c}_rgb"][k])
        P = inp[k][1] * C["PC_SCALE"] + C["PC_CENTER"]; sc.set_offsets(P[:, :2]); sc.set_facecolors(inp[k][2])
        cursor.set_xdata([k / C["FPS"]]); title.set_text(f"t = {k / C['FPS']:4.1f} s    phase: {ep['phase'][k]}")
        for t, (ans, mark, tru) in rows.items():
            ans.set_text(answers[t, k] or "(empty)")
            if (t, k) in truth:
                ok = answers[t, k] == truth[t, k]
                ans.set_color(INK); mark.set_text("✓" if ok else "✗"); mark.set_color(GOOD if ok else BAD); tru.set_text(f"truth: {truth[t, k]}")
            else:
                ans.set_color(MUTED); mark.set_text(""); tru.set_text("not graded at this frame")
            for s_, txt in brows[t].items():
                b_ = base_ans[s_][t, k] or "(empty)"; b_ = b_ if len(b_) <= 11 else b_[:10] + "…"
                if (t, k) in truth:
                    ok = base_ans[s_][t, k] == truth[t, k]; txt.set_text(f"{s_}: {b_} {'✓' if ok else '✗'}"); txt.set_color(GOOD if ok else BAD)
                else: txt.set_text(f"{s_}: {b_}"); txt.set_color(MUTED)
        so_far = lambda ans: [ans[t, kk] == truth[t, kk] for t in TASKS for kk in frames if kk <= k and (t, kk) in truth]
        done = so_far(answers)
        score.set_text(f"correct so far: fusion {sum(done)}/{len(done)}" + "".join(f"  |  {s_} {sum(so_far(v))}" for s_, v in base_ans.items())
                       + f"\nfusion runs {np.mean([cost[t] for t in TASKS]):.0%} of its encoder params per question (router)")
        return []

    stem = os.path.join(a.out, f"demo_{eid}")
    anim = animation.FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), interval=1000 * a.every / C["FPS"])
    if FFMPEG:
        anim.save(stem + "_silent.mp4", writer=animation.FFMpegWriter(fps=C["FPS"] / a.every, bitrate=3000))
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", stem + "_silent.mp4", "-i", os.path.join(d, "spoken_instruction.wav"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", stem + ".mp4"], capture_output=True)
        if r.returncode == 0: os.remove(stem + "_silent.mp4"); videos.append(stem + ".mp4")
        else: os.replace(stem + "_silent.mp4", stem + ".mp4"); print("no audio track:", r.stderr.decode()[-300:])
    else:
        anim.save(stem + ".gif", writer=animation.PillowWriter(fps=C["FPS"] / a.every))
    key = [frames[int(q * (len(frames) - 1))] for q in (0.1, 0.4, 0.65, 0.95)]; shots = []
    for k in key: draw(k); fig.canvas.draw(); shots.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    fig2, axs = plt.subplots(2, 2, figsize=(16, 9.2))
    for ax, s_ in zip(axs.ravel(), shots): ax.imshow(s_); ax.axis("off")
    plt.tight_layout(); plt.savefig(stem + "_keyframes.png", dpi=130); plt.close(fig2)

if FFMPEG and len(videos) > 1:                       # all episodes in one file (same codec settings -> stream copy)
    lst = os.path.join(a.out, "videos.txt"); open(lst, "w").write("".join(f"file '{os.path.abspath(v)}'\n" for v in videos))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(a.out, "demo_all.mp4")], check=False)
json.dump(dict(label=LABEL, use={t: list(s) for t, s in use.items()}, baselines=list(BASE), episodes=summary),
          open(os.path.join(a.out, "demo_summary.json"), "w"), indent=1)
# fusion vs baselines over all demo episodes: per question, graded frames pooled (weighted by frame count)
models = ["fusion"] + list(BASE)
def pooled(get):
    out = {}
    for t in TASKS:
        num = sum(v[0] * v[1] for e in summary.values() if (v := get(e).get(t))); den = sum(v[1] for e in summary.values() if (v := get(e).get(t)))
        if den: out[t] = num / den
    return out
pool = {"fusion": pooled(lambda e: e["accuracy"]), **{s: pooled(lambda e, s=s: e["baselines"][s]["accuracy"]) for s in BASE}}
qs = [t for t in TASKS if t in pool["fusion"]]
fig3, ax3 = plt.subplots(figsize=(1.3 * len(qs) + 2.5, 3.4), dpi=150)
pal = ["#2a78d6", "#a8a6a0", "#6250d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
bw3 = 0.8 / len(models)
for j, mname in enumerate(models):
    ax3.bar(np.arange(len(qs)) + (j - (len(models) - 1) / 2) * bw3, [pool[mname].get(t, 0) for t in qs], bw3 * 0.92,
            color=pal[j % len(pal)], label=mname, edgecolor="white", lw=0)
ax3.set_xticks(range(len(qs)), qs, fontsize=8); ax3.set_ylim(0, 1.05); ax3.set_ylabel("accuracy over demo frames (free generation)", fontsize=8)
ax3.grid(axis="y", color="0.9"); ax3.set_axisbelow(True); [ax3.spines[s].set_visible(False) for s in ("top", "right")]
ax3.legend(fontsize=7, ncol=len(models), loc="upper left", bbox_to_anchor=(0, 1.15), frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(a.out, "demo_accuracy_by_model.png")); plt.close(fig3)
print("saved", sorted(os.listdir(a.out)), flush=True)
