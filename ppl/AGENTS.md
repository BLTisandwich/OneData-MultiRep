# Contributor and coding-agent instructions

- Use English for source comments, documentation and deliverables.
- Public tracks are Action-only, QA-only and Joint-to-Action. The joint stage is internal to Joint-to-Action.
- Resolve paths through YAML/settings. Do not introduce host-specific directories, usernames or credentials.
- Keep raw observations unchanged. Put derived files under configured output/data paths.
- Preserve metadata/episodes.csv and its hash. T7 is train-only; execution tests have 209 scenarios and offline reference tests have 194 episodes.
- Keep gold, oracle goals, simulator events, expert commands and future observations out of model inputs. Use grading metadata only for labels/evaluation.
- Select checkpoints on validation only. Retain best checkpoints and complete split metrics for any reported configuration.
- Keep shared external encoder/token interfaces across models; Qwen native vision is outside the supplied comparison.
- Do not commit data, model weights, training logs, credentials or exploratory runs.
- Run `OMNIBENCH_DEVICE=cpu python -m pytest --basetemp=.test_tmp` after functional changes. CPU fixtures do not replace full-model accelerator validation.
- Document changed recipes and their practical limits. Never invent pending metrics.
