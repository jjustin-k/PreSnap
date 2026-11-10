# Benchmarks

Record validation experiments here, including data splits, preprocessing, model settings, environment, and limitations. Reserve the test set for final evaluation.

| Date | Experiment | Validation accuracy | Validation macro-F1 |
| --- | --- | ---: | ---: |
| 2026-11-08 | [Initial Random Forest](2026-11-08-random-forest-baseline.md) | 40.42% | 0.2416 |

The initial run includes `game_id` as a predictor; see the report before using it as a comparison baseline. Full-precision results are saved alongside the report as JSON.

## Hyperparameter search

[2026-11-08: Random Forest search](2026-11-08-random-forest-search/README.md) — 47 fits; selected validation macro-F1 0.3301 and accuracy 38.08% at seed 42 (three-seed mean macro-F1 0.3277). Game ID excluded, scheme-only target, both teams retained.
