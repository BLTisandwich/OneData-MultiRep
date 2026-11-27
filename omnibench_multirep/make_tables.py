"""Build results/tables.md from the run JSONs in results/ (no numbers are typed by hand).
python omnibench_multirep/make_tables.py"""
import glob, json, os
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
Q = ["task", "color", "motion", "next_action", "goal_side", "goal_dist"]
ORDER = ["blind (no inputs)", "view only", "view only (view v2)", "point only", "proprio only", "speech only", "tactile only",
         "impact only", "ours (all 4, Q-Former)", "ours (all 4, Q-Former, per-rep queries)", "ours (all 4, MLP)",
         "ours (all 4, Q-Former) + relational fusion", "ours (all 4, Q-Former) + joint attention",
         "ours (all 4, MLP) + relational fusion", "ours (all 4, Q-Former) + relational fusion, view v2",
         "ours (all 4, Q-Former) + relational fusion, view v2, asym budget", "ours (all 4, MLP), view v2, asym budget",
         "ours (all 4, Q-Former), plain training", "ours (all 4, MLP), plain training", "ours (all 6, MLP)",
         "ours (all 6, Q-Former) + relational fusion"]
f2 = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "–"

def runs(folder):
    rs = {r["setting"]: r for r in (json.load(open(f)) for f in glob.glob(os.path.join(R, folder, "*.json")))}
    return [rs[s] for s in ORDER if s in rs] + [r for s, r in rs.items() if s not in ORDER]

def main_table(rs, title):
    out = [f"### {title}", "",
           "| setting | core params (M) | " + " | ".join(Q) + " | **IID avg** | compositional avg | visual OOD avg | train min |",
           "|---|---:|" + "---:|" * len(Q) + "---:|---:|---:|---:|"]
    for r in rs:
        ood = r.get("ood", {})
        out.append(f"| {r['setting']} | {r['params_core'] / 1e6:.2f} | " + " | ".join(f2(r[q]) for q in Q)
                   + f" | **{f2(r['avg'])}** | {f2(ood.get('test_compositional', {}).get('avg'))} | "
                   f"{f2(ood.get('test_visual_ood', {}).get('avg'))} | {r['secs'] / 60:.1f} |")
    return out

def ood_table(rs):
    out = ["### Held-out splits, per question (compositional / visual OOD)", "",
           "| setting | " + " | ".join(Q) + " |", "|---|" + "---|" * len(Q)]
    for r in rs:
        o = r.get("ood", {}); c, v = o.get("test_compositional", {}), o.get("test_visual_ood", {})
        out.append(f"| {r['setting']} | " + " | ".join(f"{f2(c.get(q))} / {f2(v.get(q))}" for q in Q) + " |")
    return out

def router_tables(st):
    reps = ["view", "point", "proprio", "speech"]
    out = ["### Router (section 9 of multirep_robot.ipynb)", "",
           f"Routed model: **{st['route_setting']}** (best fusion model on the val split). Validation accuracy of the candidates: "
           + ", ".join(f"{k} {v:.3f}" for k, v in sorted(st["route_val"].items(), key=lambda x: -x[1])) + ".", "",
           "| lambda | val avg | test avg | encoder cost (share of params run) | representations / question |", "|---:|---:|---:|---:|---:|"]
    for lam, row in st["router"].items():
        star = " (lambda*)" if float(lam) == st["lambda_star"] else ""
        out.append(f"| {lam}{star} | {row['router_split']['avg']:.3f} | {row['test']['avg']:.3f} | {row['test']['cost']:.2f} | {row['test']['n_reps']:.2f} |")
    out += ["", "Final table (test_iid; the routed model with all its encoders, its best single subset, the router at lambda*, oracle routing):", "",
            "| row | " + " | ".join(Q) + " | avg | cost | reps |", "|---|" + "---:|" * (len(Q) + 3)]
    for name, v in st["final"].items():
        if "(dedicated)" in name: continue
        out.append(f"| {name.strip()} | " + " | ".join(f2(v.get(q)) for q in Q) + f" | {f2(v.get('avg'))} | {f2(v.get('cost'))} | {f2(v.get('n_reps'))} |")
    return out

def demo_table(d, split="test_iid"):
    base = d.get("baselines", [])
    out = [f"### Real-time QA demo, {split} (free greedy generation at every frame; one episode per task)", "",
           f"Fusion model: {d['label']}. Questions route to: " + "; ".join(f"{t} -> {'+'.join(s)}" for t, s in d["use"].items()) + ".", "",
           "| episode | task | fusion | " + " | ".join(base) + " |", "|---|---|---:|" + "---:|" * len(base)]
    for eid, e in d["episodes"].items():
        out.append(f"| {eid} | {e['task']} | **{f2(e['overall'])}** | " + " | ".join(f2(e["baselines"][b]["overall"]) for b in base) + " |")
    return out

def shortcut_table(path):
    if not os.path.exists(path): return []
    s = json.load(open(path)); rows = s["shortcut_frame"]
    cols = ["none", "task", "time", "task+time"]
    out = ["### Shortcut baselines of the frame questions (test_iid; majority answer of train given ...)", "",
           "| question | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for q in Q:
        if q in rows: out.append(f"| {q} | " + " | ".join(f2(rows[q].get(f"test_iid/{c}")) for c in cols) + " |")
    return out

if __name__ == "__main__":
    md = ["# OmniBench frame QA with the multi-representation adaptor (Qwen2.5-0.5B-Instruct, frozen)", "",
          "Generated by `make_tables.py` from the JSON files in this folder. One training seed; 2M trainable parameters per "
          "setting (encoders + fusion + aggregator; LLM-interface layers reported separately).", ""]
    md += shortcut_table(os.path.join(R, "cache_summary.json")) + [""]
    md += main_table(runs("runs"), "All 678 aligned episodes, 8 tasks (main run)") + [""]
    md += ood_table(runs("runs")) + [""]
    md += router_tables(json.load(open(os.path.join(R, "router_stages.json")))) + [""]
    if os.path.exists(os.path.join(R, "router_choices.md")): md += open(os.path.join(R, "router_choices.md")).read().splitlines() + [""]
    for split in ("test_iid", "test_compositional", "test_visual_ood"):
        p = os.path.join(R, "demo_summary.json" if split == "test_iid" else f"demo_summary_{split}.json")
        if os.path.exists(p): md += demo_table(json.load(open(p)), split) + [""]
    md += main_table(runs("lee_pilot"), "Pilot: Lee shard only (400 raw episodes, T1/T3/T5/T8), key settings") + [""]
    open(os.path.join(R, "tables.md"), "w", encoding="utf8", newline="\n").write("\n".join(md))
    print("wrote", os.path.join(R, "tables.md"))
