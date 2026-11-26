# Evaluation protocol

## EE prediction

Use accepted aligned reference test episodes only (61 IID, 67 compositional, 66 visual OOD). Evaluate all 95 valid causal action examples per episode, preserving episode boundaries. Report world-frame absolute EE-target MAE and relative-delta MAE in centimeters, next observed EE-position error, gripper accuracy/F1, and per-task and per-split breakdowns. The absolute and delta errors coincide when both use the same current observed EE position.

## Grounded QA and encoder evidence

Questions name observable properties rather than merely asking for a task label. Programmatic answers retain input-branch identity, answer type, numeric tolerance, supporting fields and time ranges. Candidate selection rules fit the **training split only** and remove task/question contexts that admit a strong constant-answer shortcut. Not every question needs every modality or multiple modalities.

The evaluator generates answers and grades four conditions:

| Condition | Input | Expected answer |
|---|---|---|
| `full_matched` | All sensor branches | Original grounded answer |
| `isolated_matched` | Question-associated branch(es) | Original grounded answer |
| `isolated_zero` | Associated feature representations replaced by zero | Original answer, for a reliance control |
| `isolated_swapped` | Features from a different episode, matched on task/split/question context | Donor episode's grounded answer |

Swapped controls use donors with different grounded answers and grade against the donor. Donor answers and evidence fields never enter the prompt. Donor selection stays within the same split; it is an evaluation intervention, not cross-split training. Compare conditions to assess whether the model uses encoded information; full-observation QA accuracy alone cannot establish modality reliance.

QA macro accuracy averages the available encoder/question categories; per-category counts are retained because some force/audio/tactile questions are sparse. Numerical answers use the saved tolerance. Discrete answers must be unambiguous: an answer listing competing colors or both yes/no is rejected. Answer generation is greedy and limited to 12 tokens by default.

Joint-to-Action evaluation additionally removes each input family from the same trained checkpoint: visual, 3D, speech/language, environmental audio, time series, force, tactile, and proprioception. These are inference-time removals, not retrained ablations. Correlated event-derived proxies can make some individual removals inconclusive.

## Actual execution

The test manifest contains 209 original scenarios: 69 IID, 70 compositional and 70 visual OOD. T7 is excluded. The evaluator starts from each original scene and runs 96 actual model commands. It renders fresh observations, recomputes joints/contacts/proxies from executed state and never feeds the expert phase schedule or future commands into the model.

Each predicted XYZ delta is limited to ±0.10 m per axis and the EE target is clipped to the simulator workspace. Gripper predictions are mapped to finger openings. Task interaction assistance is based on current state: assisted attachment/motion and virtual press/tap triggers are retained from the declared simulator protocol.

Success requires the source task-goal predicate and, for placement tasks T1/T3/T4, an observed release event and no attachment at the end. Penetration/height flags are diagnostics only and do not themselves cause failure. This is **assisted kinematic simulation completion**, not physical force-closure or real-robot success. Expert reference success is never substituted for model execution success.

## Outputs and uncertainty

`run` automatically evaluates each validation-selected public checkpoint. Offline predictions and execution records share the checkpoint hash and partition hash. Results include per-split/task metrics. EE and QA confidence intervals use episode-cluster bootstrap resampling; newly generated execution summaries use Wilson 95% intervals. Historical selected execution JSON retains its original episode-bootstrap intervals. Do not compare these two interval methods as if they were identical.

The pure Action-only track has EE/execution results. QA-only has generated QA results. Joint-to-Action has both plus modality interventions; its final QA measures retention after action specialization. Keep all splits for any reported configuration and label the selected track. Do not describe separate Action-only and QA-only weights as one jointly competent checkpoint.
