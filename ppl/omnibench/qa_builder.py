"""QA-only OmniBench optimization: deterministic labels, train-only shortcut filtering.
No source sensor files are modified. No network calls or API keys are used.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys, time, wave
from pathlib import Path
import numpy as np
from .questions import make_questions, VERSION
from .settings import ROOT, path
PROJECT = ROOT
NUMERIC_TOL = 1e-6


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def audio_info(path, detect_events=False):
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Expected 16-bit PCM WAV")
        signal = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    signal = signal.astype(np.float64).reshape(-1, channels).mean(axis=1) / 32768
    result = {"sample_rate": rate, "channels": channels,
              "duration_s": len(signal) / rate, "nonempty": bool(signal.size),
              "nonzero": bool(np.any(signal)), "peak_abs": float(np.max(abs(signal))),
              "rms": float(np.sqrt(np.mean(signal ** 2))), "bursts": []}
    if detect_events and np.any(signal):
        hop = max(1, round(rate * .01))
        n = len(signal) // hop
        energy = np.sqrt(np.mean(signal[:n * hop].reshape(n, hop) ** 2, axis=1))
        active = np.flatnonzero(energy > max(1e-5, energy.max() * .10))
        groups = np.split(active, np.flatnonzero(np.diff(active) > 3) + 1)
        result["bursts"] = [[float(g[0] * hop / rate), float((g[-1] + 1) * hop / rate)]
                            for g in groups if len(g)]
    return result

def reviewed_v1(original, state, physics, events):
    """Keep original gold separately; make source and thresholds explicit here."""
    revised = []
    for item in original:
        q = dict(item)
        q["protocol"] = "canonical-v1-reviewed"
        q["original_answer"] = item["expert_answer"]
        q["original_question"] = item["question"]
        category = q["category"]
        if category == "outcome":
            q["question"] = "Does the object's last recorded height exceed its first recorded height by more than 2 centimeters?"
            z = state["object_position_world_m"][:, 2]
            q["expert_answer"] = "yes" if z[-1] - z[0] > .02 else "no"
            q["definition_change"] = "Explicit 2 cm meaningful-height threshold; original wording was underspecified."
            q["evidence_time_range_s"] = [float(state["timestamp_s"][0]), float(state["timestamp_s"][-1])]
        elif category == "cross_modal":
            q["question"] = "Does the recorded 240 Hz robot-object contact signal exceed 0.000001 N anywhere in the eight-second episode?"
            q["expert_answer"] = "yes" if np.any(physics["robot_object_contact_N"] > NUMERIC_TOL) else "no"
            q["answer_source"] = "physics_240hz.npz:robot_object_contact_N; complete 240 Hz episode"
            q["evidence_time_range_s"] = [float(physics["time_s"][0]), float(physics["time_s"][-1])]
        elif category == "temporal" and events:
            stamp = min(e["time_s"] for e in events)
            q["evidence_time_range_s"] = [stamp, stamp]
        q.pop("api_paraphrase", None)
        q.pop("api_paraphrase_verified", None)
        revised.append(q)
    return revised

def policy_key(q):
    # Include the exact question/time window so fixed script timing cannot hide
    # behind a mixture of questions with different answers.
    return json.dumps([q["task_id"], q["template_id"], q["question"]], ensure_ascii=False)

def constant_prediction(qs):
    if qs[0]["answer_type"] == "number":
        values = sorted(float(q["expert_answer"]) for q in qs)
        tol = qs[0]["numeric_tolerance"]
        left, best = 0, (0, 0)
        for right, value in enumerate(values):
            while value - values[left] > 2 * tol + 1e-10:
                left += 1
            if right - left + 1 > best[1] - best[0] + 1:
                best = (left, right)
        prediction = (values[best[0]] + values[best[1]]) / 2
        score = sum(abs(v - prediction) <= tol + 1e-10 for v in values) / len(values)
    else:
        counts = collections.Counter(q["expert_answer"] for q in qs)
        prediction = sorted(counts, key=lambda x: (-counts[x], x))[0]
        score = counts[prediction] / len(qs)
    return prediction, score

def correct(q, predicted):
    if q["answer_type"] == "number":
        return abs(float(q["expert_answer"]) - float(predicted)) <= q["numeric_tolerance"] + 1e-10
    return q["expert_answer"] == str(predicted)


def discover(sources, sample_per_task=0):
    result = []
    for owner, root in sources.items():
        manifest = json.loads((root / "task_manifest.json").read_text())
        groups = collections.defaultdict(list)
        for row in manifest["rows"]:
            eid = row["episode_id"]
            if Path(eid).name != eid:
                raise ValueError("Invalid episode ID")
            groups[row["task_id"]].append({**row, "source_key": owner, "source_path": root / "episodes" / eid})
        for rows in groups.values():
            rows.sort(key=lambda x: x["episode_id"])
            if sample_per_task:
                indices = np.linspace(0, min(len(rows) - 1, int(len(rows) * .8)), sample_per_task).astype(int)
                rows = [rows[i] for i in sorted(set(indices))]
            result.extend(rows)
    return sorted(result, key=lambda x: x["episode_id"])


def build_episode(item):
    p = item["source_path"]
    task = json.loads((p / "task.json").read_text())
    state = load_npz(p / "robot_object_state.npz")
    physics = load_npz(p / "physics_240hz.npz")
    tactile = load_npz(p / "tactile_proxy.npz")
    for source in [state, physics, tactile]:
        for a in source.values():
            if a.dtype.kind in "fc" and not np.isfinite(a).all():
                raise ValueError("Nonfinite label source: " + item["episode_id"])
    audio = audio_info(p / "impact_audio_SYNTHETIC.wav", detect_events=True)
    if not audio["nonempty"]:
        raise ValueError("Empty environmental WAV")
    events = json.loads((p / "events.json").read_text())
    original = [json.loads(x) for x in (p / "qa.jsonl").read_text().splitlines()]
    reviewed = reviewed_v1(original, state, physics, events)
    audio['speech_duration_s'] = audio_info(p / 'spoken_instruction.wav')['duration_s']
    candidates = make_questions(p, task, state, physics, tactile, audio)
    minimum_z = float(physics["object_xyz_world_m"][:, 2].min())
    severe = minimum_z < -.01 or physics["object_xyz_world_m"][-1, 2] < 0
    original_lookup = []
    for q in original:
        original_lookup.append({**q, "task_id": task["task_id"], "split": task["split"],
                                "template_id": q["category"], "numeric_tolerance": None})
    for i, q in enumerate(candidates):
        q.update(question_id=f"{item['episode_id']}_V3_{i + 1:02d}", episode_id=item["episode_id"],
                 task_id=task["task_id"], split=task["split"])
    for q in reviewed:
        q.update(episode_id=item["episode_id"], task_id=task["task_id"], split=task["split"])
    record = {"episode_id": item["episode_id"], "owner": item["source_key"], "task_id": task["task_id"],
              "split": task["split"], "seed": task["seed"], "source_episode": "episodes/" + item["episode_id"],
              "primary_qa_eligible": not severe, "minimum_object_world_z_m": minimum_z,
              "quality_flag": "severe_geometry_review_required" if severe else
                              "minor_negative_z_review" if minimum_z < 0 else None,
              "v1_answer_changes": sum(q["expert_answer"] != q["original_answer"] for q in reviewed),
              "source_checksum_manifest_sha256": hashlib.sha256((p / "_COMPLETE.json").read_bytes()).hexdigest(),
              "original": original_lookup, "reviewed": reviewed, "candidates": candidates}
    return record


def fit_rules(qs, minimum=5):
    groups = collections.defaultdict(list)
    for q in qs:
        if q["split"] == "train":
            groups[policy_key(q)].append(q)
    rules = {}
    for key, group in groups.items():
        prediction, accuracy = constant_prediction(group)
        rules[key] = {"train_examples": len(group), "constant_prediction": prediction,
                      "train_constant_accuracy": accuracy,
                      "primary": len(group) >= minimum and accuracy < .9}
    return rules


def baseline(qs, rules):
    result = {}
    for split in ["train", "val", "test_iid", "test_compositional", "test_visual_ood"]:
        subset = [q for q in qs if q["split"] == split]
        covered = [q for q in subset if policy_key(q) in rules]
        good = sum(correct(q, rules[policy_key(q)]["constant_prediction"]) for q in covered)
        result[split] = {"questions": len(subset), "covered": len(covered),
                         "accuracy": good / len(covered) if covered else None}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="OWNER=relative/owner_folder; repeat for both owners")
    parser.add_argument("--output", type=Path, default=path("qa"))
    parser.add_argument("--sample-per-task", type=int, default=0)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.source:
        sources = {x.split("=", 1)[0]: Path(x.split("=", 1)[1]).resolve() for x in args.source}
    else:
        sources = {owner: PROJECT / "data" / owner for owner in ["Lee", "Lazizbek"]
                   if (PROJECT / "data" / owner / "task_manifest.json").exists()}
    if not sources:
        raise ValueError("No source manifests found")
    for root in sources.values():
        if args.output == root or args.output.is_relative_to(root) or root.is_relative_to(args.output):
            raise ValueError("Output must be separate from original data folders")
    items = discover(sources, args.sample_per_task)
    records = []
    for i, item in enumerate(items, 1):
        records.append(build_episode(item))
        if i % 100 == 0 or i == len(items):
            print(f"Read QA label sources: {i}/{len(items)}", flush=True)
    train_cohort = [r for r in records if r["primary_qa_eligible"]]
    rules = fit_rules([q for r in train_cohort for q in r["candidates"]])
    old_questions = [q for r in train_cohort for q in r["original"]]
    old_rules = fit_rules(old_questions)
    primary, auxiliary, reviewed, inputs, summaries = [], [], [], [], []
    for record in records:
        kept = []
        for q in record["candidates"]:
            rule = rules.get(policy_key(q))
            use = record["primary_qa_eligible"] and q['primary_candidate'] and bool(rule and rule["primary"])
            q["track"] = "primary" if use else "auxiliary"
            q["selection_reason"] = "variable_training_context" if use else (
                "severe_geometry_review" if not record["primary_qa_eligible"] else
                "insufficient_train_context" if not rule or rule["train_examples"] < 5 else
                "task_question_constant_shortcut")
            (primary if use else auxiliary).append(q)
            if use:
                kept.append(q)
                inputs.append({k: q[k] for k in ["episode_id", "question_id", "question", "encoder_branch", "input_modalities", "input_mode", "input_fields"]})
        reviewed.extend(record["reviewed"])
        row = {k: v for k, v in record.items() if k not in ["original", "reviewed", "candidates"]}
        row["primary_question_count"] = len(kept)
        row["auxiliary_question_count"] = len(record["candidates"]) - len(kept)
        summaries.append(row)
        dump(args.output / "per_episode" / (record["episode_id"] + ".json"),
             {"metadata": row, "primary": kept, "auxiliary": [q for q in record["candidates"] if q["track"] == "auxiliary"],
              "canonical_v1_reviewed": record["reviewed"]})
    jsonl(args.output / "qa_primary.jsonl", primary)
    jsonl(args.output / "qa_auxiliary.jsonl", auxiliary)
    jsonl(args.output / "qa_v1_reviewed.jsonl", reviewed)
    jsonl(args.output / "qa_model_inputs.jsonl", inputs)
    jsonl(args.output / "episodes.jsonl", summaries)
    # Same-task, same-split, same-question replacement prevents task/time priors
    # from explaining a feature intervention. Gold stays in grade-only manifests.
    groups = collections.defaultdict(list)
    for q in primary:
        groups[(q['split'], policy_key(q))].append(q)
    controls = []
    for q in primary:
        peers = groups[(q['split'], policy_key(q))]
        donors = [p for p in peers if p['episode_id'] != q['episode_id'] and
                  (abs(float(p['expert_answer'])-float(q['expert_answer'])) > 2*q['numeric_tolerance']
                   if q['answer_type'] == 'number' else p['expert_answer'] != q['expert_answer'])]
        if not donors:
            continue
        donor = sorted(donors, key=lambda p: hashlib.sha256((q['question_id']+p['question_id']).encode()).hexdigest())[0]
        controls.append(dict(question_id=q['question_id'], donor_question_id=donor['question_id'],
                             episode_id=q['episode_id'], donor_episode_id=donor['episode_id'],
                             task_id=q['task_id'], split=q['split'], encoder_branch=q['encoder_branch'],
                             original_answer=q['expert_answer'], swapped_expected_answer=donor['expert_answer'],
                             interventions=['matched_features','zero_features','same_task_same_split_swapped_features']))
    jsonl(args.output / 'feature_controls_grade_only.jsonl', controls)
    jsonl(args.output / 'feature_controls_inputs.jsonl',
          [{k:v for k,v in c.items() if k not in ['original_answer','swapped_expected_answer']} for c in controls])
    dump(args.output / "selection_rules.json", {"fit_split": "train", "minimum_context_examples": 5,
         "constant_shortcut_cutoff": .9, "rules": rules})
    report = {"version": VERSION, "episodes_read": len(records),
              "primary_eligible_episodes": sum(r["primary_qa_eligible"] for r in records),
              "severe_geometry_flagged_episodes": [r["episode_id"] for r in records if not r["primary_qa_eligible"]],
              "primary_qa": len(primary), "auxiliary_qa": len(auxiliary), "reviewed_v1_qa": len(reviewed),
              "feature_control_pairs": len(controls),
              "encoder_branch_counts": dict(collections.Counter(q['encoder_branch'] for q in primary)),
              "v1_answer_corrections": sum(r["v1_answer_changes"] for r in records),
              "primary_categories": dict(collections.Counter(q["category"] for q in primary)),
              "primary_by_task": dict(collections.Counter(q["task_id"] for q in primary)),
              "primary_by_split": dict(collections.Counter(q["split"] for q in primary)),
              "old_v1_task_question_lookup": baseline(old_questions, old_rules),
              "new_primary_task_question_lookup": baseline(primary, rules),
              "api_calls": 0, "api_cost_usd": 0, "source_sensor_files_modified": False,
              "mode": "smoke_test_only" if args.sample_per_task else "full_qa_export",
              "notes": ["Train-only constant shortcut filtering; held-out answers do not select questions.",
                        "Numerical QA uses published tolerances; do not compare accuracy to old categorical QA as a model improvement.",
                        "All valid synthetic/proxy sensor streams are retained in original data. Only QA is changed.",
                        "The original QA files remain untouched; reviewed v1 is a separate protocol with explicit definitions.",
                        "Isolated encoder probes use explicit field allowlists. TTS transcript, oracle object states and answer sources are grade-only.",
                        "Feature controls are prepared; no encoder-use conclusion is claimed before actual matched/zero/swapped model evaluation.",
                        "This export does not certify closed-loop execution or repair physics trajectories."]}
    dump(args.output / "report.json", report)
    dump(args.output / "dataset_manifest.json", {"version": VERSION, "qa_only": True,
         "source_roots": {k: str(v) for k, v in sources.items()},
         "split_policy": "All derived questions inherit the original episode split; no frames or paraphrases cross splits.",
         "input_protocol": "Use qa_model_inputs.jsonl for prompts; load sensor observations from source_roots. Do not concatenate label/provenance metadata.",
         "source_episode_counts": dict(collections.Counter(r["owner"] for r in records)),
         "eligible_episode_counts_by_split": dict(collections.Counter(r["split"] for r in records if r["primary_qa_eligible"])),
         "episodes": summaries})
    dump(args.output / "checksums.json", {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
