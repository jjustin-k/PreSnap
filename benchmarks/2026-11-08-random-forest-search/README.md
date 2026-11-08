# Random Forest hyperparameter search — 2026-11-08

Completed 41 candidate configurations and six seed-confirmation runs (47 fits). Selected 300 trees, depth 20, leaf size 1, log2 feature sampling, and balanced class weights. The trainer now uses these settings with seed 42.

| Configuration | Validation macro-F1 | Validation accuracy |
| --- | ---: | ---: |
| Unweighted defaults, seed 42 | 0.2479 | 39.56% |
| Previous balanced forest, seed 42 | 0.3198 | 35.73% |
| Selected configuration, seed 42 | 0.3301 | 38.08% |
| Selected configuration, three-seed mean | 0.3277 | 37.58% |

The seed-42 improvement over the previous balanced model is modest: 0.0103 macro-F1 and 2.35 accuracy percentage points. Baselines were run only at seed 42; statistical significance has not been established.

## Why these settings were tested

1. **Class weights: None, balanced, balanced_subsample.** Compare unweighted learning with emphasizing rare classes, using either full-training or per-bootstrap class frequencies.
2. **Depth: None, 10, 20.** Test whether limiting complexity reduces memorization; unrestricted depth is the control.
3. **Minimum leaf samples: 1, 3, 5, 10.** Test whether requiring more observations per leaf improves generalization. Crossed with weights and depth for 36 runs.
4. **Features per split: sqrt, log2, 0.5, all.** On the best grid configuration, test the tradeoff between tree diversity and considering more feature interactions. Counts apply after one-hot encoding.
5. **Trees: 100, 300, 500.** On the best configuration so far, test whether extra averaging improves stability and predictions.
6. **Seeds: 42, 7, 123.** Recheck the top three candidates at two additional seeds. Select highest mean validation macro-F1 across three seeds, breaking ties by mean accuracy. Data partitions stay fixed.

This is a staged search, not an exhaustive cross-product. Other classifier settings remain defaults (including Gini criterion, bootstrapping, and minimum split size 2). Larger leaves generally hurt the balanced model; feature sampling and moderate depth limits helped the strongest candidates.

## Evaluation controls

- Training: 5,921 plays / 85 games. Validation: 1,279 plays / 18 games. Reserved test: 1,315 plays / 19 games.
- Same two GroupShuffleSplit operations as the trainer, both seeded 42; game partitions verified disjoint.
- Game ID excluded from predictors. Both offensive and defensive teams retained. Target is the coverage scheme alone.
- OneHotEncoder(handle_unknown="ignore") fitted only on training data; numeric features passed through. Encoded matrices reused across runs.
- Metrics use a consistent list of 12 classes. Per-class reports, training scores, timings, parameters, and reasons recorded for every run.
- No test predictions or test metrics computed. Only test partition size and game overlap checked.
- summary.json records class counts, majority baseline, feature names, environment versions, and SHA-256 fingerprints. The trainer fingerprint captures its pre-update contents.

## Seed sensitivity

| Initial run | Run IDs across seeds | Mean macro-F1 | Population standard deviation | Mean accuracy |
| --- | --- | ---: | ---: | ---: |
| 40 | 40, 44, 45 | 0.3277 | 0.0066 | 0.3758 |
| 37 | 37, 42, 43 | 0.3135 | 0.0134 | 0.3563 |
| 21 | 21, 46, 47 | 0.3077 | 0.0236 | 0.3591 |

Run 37 had the highest seed-42 macro-F1 (0.3318), but run 40 had the highest three-seed mean and lower seed variability. These repeats measure algorithm randomness, not uncertainty across new games.

## Selected configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=1,
    max_features='log2',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
```

## Selected model classification report (seed 42)

| Class / average | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 2-Man | 0.134 | 0.200 | 0.161 | 45 |
| Bracket | 0.000 | 0.000 | 0.000 | 8 |
| Cover-0 | 0.588 | 0.159 | 0.250 | 63 |
| Cover-1 | 0.506 | 0.491 | 0.498 | 326 |
| Cover-2 | 0.246 | 0.270 | 0.257 | 152 |
| Cover-3 | 0.517 | 0.328 | 0.401 | 375 |
| Cover-6 | 0.217 | 0.371 | 0.274 | 89 |
| Goal Line | 0.750 | 1.000 | 0.857 | 6 |
| Miscellaneous | 0.000 | 0.000 | 0.000 | 2 |
| Prevent | 0.200 | 0.500 | 0.286 | 2 |
| Quarters | 0.268 | 0.370 | 0.311 | 162 |
| Red Zone | 0.530 | 0.898 | 0.667 | 49 |
| macro avg | 0.330 | 0.382 | 0.330 | 1279 |
| weighted avg | 0.417 | 0.381 | 0.382 | 1279 |

## All runs

None means unweighted under weights, unrestricted under depth, and all features under feature sampling. Full precision is retained in JSON.

| Run | Stage | Weights | Depth | Leaf size | Features | Trees | Seed | Train macro-F1 | Val macro-F1 | Val accuracy |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | weight-depth-leaf | None | None | 1 | sqrt | 100 | 42 | 1.0000 | 0.2479 | 0.3956 |
| 2 | weight-depth-leaf | None | None | 3 | sqrt | 100 | 42 | 0.5724 | 0.2184 | 0.4003 |
| 3 | weight-depth-leaf | None | None | 5 | sqrt | 100 | 42 | 0.4233 | 0.2117 | 0.4058 |
| 4 | weight-depth-leaf | None | None | 10 | sqrt | 100 | 42 | 0.2173 | 0.1525 | 0.4050 |
| 5 | weight-depth-leaf | None | 10 | 1 | sqrt | 100 | 42 | 0.4042 | 0.1975 | 0.3901 |
| 6 | weight-depth-leaf | None | 10 | 3 | sqrt | 100 | 42 | 0.2914 | 0.2003 | 0.3831 |
| 7 | weight-depth-leaf | None | 10 | 5 | sqrt | 100 | 42 | 0.2483 | 0.1921 | 0.3839 |
| 8 | weight-depth-leaf | None | 10 | 10 | sqrt | 100 | 42 | 0.1687 | 0.1342 | 0.3831 |
| 9 | weight-depth-leaf | None | 20 | 1 | sqrt | 100 | 42 | 0.9175 | 0.2222 | 0.4019 |
| 10 | weight-depth-leaf | None | 20 | 3 | sqrt | 100 | 42 | 0.4848 | 0.2162 | 0.4058 |
| 11 | weight-depth-leaf | None | 20 | 5 | sqrt | 100 | 42 | 0.3709 | 0.2117 | 0.4058 |
| 12 | weight-depth-leaf | None | 20 | 10 | sqrt | 100 | 42 | 0.2049 | 0.1467 | 0.3995 |
| 13 | weight-depth-leaf | balanced | None | 1 | sqrt | 100 | 42 | 0.9620 | 0.3198 | 0.3573 |
| 14 | weight-depth-leaf | balanced | None | 3 | sqrt | 100 | 42 | 0.5880 | 0.2883 | 0.3425 |
| 15 | weight-depth-leaf | balanced | None | 5 | sqrt | 100 | 42 | 0.4690 | 0.2633 | 0.3026 |
| 16 | weight-depth-leaf | balanced | None | 10 | sqrt | 100 | 42 | 0.3541 | 0.2523 | 0.2838 |
| 17 | weight-depth-leaf | balanced | 10 | 1 | sqrt | 100 | 42 | 0.5165 | 0.2832 | 0.2862 |
| 18 | weight-depth-leaf | balanced | 10 | 3 | sqrt | 100 | 42 | 0.3895 | 0.2539 | 0.2768 |
| 19 | weight-depth-leaf | balanced | 10 | 5 | sqrt | 100 | 42 | 0.3513 | 0.2436 | 0.2737 |
| 20 | weight-depth-leaf | balanced | 10 | 10 | sqrt | 100 | 42 | 0.2782 | 0.2101 | 0.2447 |
| 21 | weight-depth-leaf | balanced | 20 | 1 | sqrt | 100 | 42 | 0.8706 | 0.3284 | 0.3737 |
| 22 | weight-depth-leaf | balanced | 20 | 3 | sqrt | 100 | 42 | 0.5382 | 0.2813 | 0.3276 |
| 23 | weight-depth-leaf | balanced | 20 | 5 | sqrt | 100 | 42 | 0.4378 | 0.2778 | 0.3112 |
| 24 | weight-depth-leaf | balanced | 20 | 10 | sqrt | 100 | 42 | 0.3380 | 0.2486 | 0.2729 |
| 25 | weight-depth-leaf | balanced_subsample | None | 1 | sqrt | 100 | 42 | 1.0000 | 0.2427 | 0.3995 |
| 26 | weight-depth-leaf | balanced_subsample | None | 3 | sqrt | 100 | 42 | 0.7945 | 0.3256 | 0.3823 |
| 27 | weight-depth-leaf | balanced_subsample | None | 5 | sqrt | 100 | 42 | 0.6279 | 0.3077 | 0.3628 |
| 28 | weight-depth-leaf | balanced_subsample | None | 10 | sqrt | 100 | 42 | 0.4594 | 0.2851 | 0.3339 |
| 29 | weight-depth-leaf | balanced_subsample | 10 | 1 | sqrt | 100 | 42 | 0.6460 | 0.2862 | 0.3073 |
| 30 | weight-depth-leaf | balanced_subsample | 10 | 3 | sqrt | 100 | 42 | 0.4780 | 0.2831 | 0.2979 |
| 31 | weight-depth-leaf | balanced_subsample | 10 | 5 | sqrt | 100 | 42 | 0.4256 | 0.2728 | 0.3190 |
| 32 | weight-depth-leaf | balanced_subsample | 10 | 10 | sqrt | 100 | 42 | 0.3417 | 0.2458 | 0.2752 |
| 33 | weight-depth-leaf | balanced_subsample | 20 | 1 | sqrt | 100 | 42 | 0.9798 | 0.3035 | 0.3698 |
| 34 | weight-depth-leaf | balanced_subsample | 20 | 3 | sqrt | 100 | 42 | 0.7001 | 0.3242 | 0.3683 |
| 35 | weight-depth-leaf | balanced_subsample | 20 | 5 | sqrt | 100 | 42 | 0.5766 | 0.2992 | 0.3495 |
| 36 | weight-depth-leaf | balanced_subsample | 20 | 10 | sqrt | 100 | 42 | 0.4322 | 0.2872 | 0.3190 |
| 37 | feature-sampling | balanced | 20 | 1 | log2 | 100 | 42 | 0.8796 | 0.3318 | 0.3628 |
| 38 | feature-sampling | balanced | 20 | 1 | 0.5 | 100 | 42 | 0.8515 | 0.3100 | 0.3370 |
| 39 | feature-sampling | balanced | 20 | 1 | None | 100 | 42 | 0.8066 | 0.2924 | 0.2995 |
| 40 | tree-count | balanced | 20 | 1 | log2 | 300 | 42 | 0.8905 | 0.3301 | 0.3808 |
| 41 | tree-count | balanced | 20 | 1 | log2 | 500 | 42 | 0.8950 | 0.3227 | 0.3714 |
| 42 | seed-check | balanced | 20 | 1 | log2 | 100 | 7 | 0.8792 | 0.3085 | 0.3526 |
| 43 | seed-check | balanced | 20 | 1 | log2 | 100 | 123 | 0.8801 | 0.3002 | 0.3534 |
| 44 | seed-check | balanced | 20 | 1 | log2 | 300 | 7 | 0.8938 | 0.3187 | 0.3604 |
| 45 | seed-check | balanced | 20 | 1 | log2 | 300 | 123 | 0.8915 | 0.3343 | 0.3862 |
| 46 | seed-check | balanced | 20 | 1 | sqrt | 100 | 7 | 0.8887 | 0.3201 | 0.3565 |
| 47 | seed-check | balanced | 20 | 1 | sqrt | 100 | 123 | 0.8806 | 0.2747 | 0.3471 |

## Limitations

Selection over many runs can overfit one validation split. Goal Line has six validation examples; Prevent and Miscellaneous each have two, making macro-F1 sensitive to a handful of predictions. Grouped cross-validation within training games would provide a useful additional check before final test evaluation. The selected set is best among the configurations tested under the stated selection rule, not a proven global optimum.

## Reproduce

```bash
venv/bin/python scripts/tune_random_forest.py
```

Run from the repository root with the recorded dataset and library versions. The script overwrites this experiment directory’s JSON records; preserve a copy before rerunning on changed data. This Markdown report is a snapshot and is not regenerated automatically.

- [All runs with reasons and per-class results](runs.json)
- [Summary, fingerprints, split metadata, and finalists](summary.json)
- Search script: `scripts/tune_random_forest.py`
