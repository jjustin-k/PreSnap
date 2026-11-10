# Initial Random Forest benchmark — 2026-11-08

## Result

Re-ran the current working-tree training code against the existing cleaned CSV and exactly reproduced the user-provided screenshot's accuracy and macro-F1. This is a **validation benchmark**, not a final test result. Training code and input data were not modified.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Random Forest | 0.4042220484753714 | 0.2415970249389852 |
| Training-majority baseline (`Cover-3-Zone`) | 0.2931978107896794 | 0.0377871825876663 |

The majority baseline was fitted on training labels and evaluated on the same validation rows. The forest improves accuracy by 11.10 percentage points.

## Data and split

- Input: `data/cleaned_data/cleaned_data.csv` (8,515 plays across 122 games).
- Target: `defensive_coverage`, concatenating `pff_passCoverage` and `pff_passCoverageType`.
- Ingestion joins labels on `(gameId, playId)`, converts the quarter clock to seconds, computes absolute score gap and distance to goal, and extracts personnel counts.
- Two `GroupShuffleSplit` operations: 70% training games, then half the remaining games for validation and half for test; both use `random_state=42`.
- Training: 5,921 plays / 85 games. Validation: 1,279 plays / 18 games. Test: 1,315 plays / 19 games.
- Verified no game overlap between partitions. Test predictions and metrics were not computed.

## Configuration

Source: `presnap/training/random_forest_trainer.py` and `presnap/pipeline/ingest.py`.

Categorical columns use `OneHotEncoder(handle_unknown="ignore")`; remaining columns pass through unchanged. Preprocessing is fitted within the training pipeline.

`RandomForestClassifier(random_state=42, n_jobs=-1)` otherwise uses defaults: 100 trees, Gini criterion, square-root feature sampling, bootstrapping enabled, unlimited depth, minimum split size 2, minimum leaf size 1, and no class weighting.

Input columns (before encoding):

```text
game_id, offensive_team, defensive_team, quarter, game_clock, score_gap, down, yards_to_fd, yards_to_goal, offensive_line, wide_recievers, running_backs, tight_ends, formation, defensive_line, line_backers, defensive_backs, defenders_in_box
```

Environment: Python 3.14.2, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4.

## Classification report

Values below are rounded to two decimals, matching the screenshot. Full precision, complete classifier parameters, and SHA-256 fingerprints of both source files and the input CSV are in [the JSON record](2026-11-08-random-forest-baseline.json). The fingerprints capture the working-tree contents, including uncommitted edits.

| Coverage / average | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 2-Man-Man | 0.00 | 0.00 | 0.00 | 45 |
| Bracket-Other | 0.00 | 0.00 | 0.00 | 8 |
| Cover-0-Man | 0.57 | 0.06 | 0.11 | 63 |
| Cover-1-Man | 0.45 | 0.50 | 0.48 | 326 |
| Cover-2-Zone | 0.30 | 0.14 | 0.19 | 152 |
| Cover-3-Zone | 0.39 | 0.69 | 0.49 | 375 |
| Cover-6-Zone | 0.10 | 0.02 | 0.04 | 89 |
| Goal Line-Other | 0.80 | 0.67 | 0.73 | 6 |
| Miscellaneous-Other | 0.00 | 0.00 | 0.00 | 2 |
| Prevent-Other | 0.00 | 0.00 | 0.00 | 2 |
| Quarters-Zone | 0.33 | 0.15 | 0.21 | 162 |
| Red Zone-Other | 0.54 | 0.84 | 0.66 | 49 |
| macro avg | 0.29 | 0.26 | 0.24 | 1279 |
| weighted avg | 0.37 | 0.40 | 0.35 | 1279 |

## Interpretation and limitations

- **`game_id` is currently a predictor.** Training drops only the target column, so the game ID passes through as a numeric feature. Game partitions are disjoint, but this identifier may encode incidental date/order patterns. Preserve this result as the original benchmark; remove the ID from predictors and record a separate run before tuning.
- Cover-1 and Cover-3 have the strongest F1 among the common coverage classes. Cover-3 recall exceeds precision, indicating many false-positive Cover-3 predictions.
- Four classes have zero recall. Goal Line's relatively high F1 rests on only six validation examples; rare-class scores are unstable.
- Current inputs describe game context and personnel rather than detailed player-tracking alignment.
- Ingestion assumes fixed personnel string positions and one quarterback when deriving offensive-line counts. It skips invalid/missing records without a structured exclusion summary. This benchmark does not establish ingestion correctness for every source row.

## Reproduce

From the repository root, with the recorded input CSV and compatible dependencies:

```bash
venv/bin/python presnap/training/random_forest_trainer.py
```

The command prints validation metrics and opens a normalized confusion matrix. For a noninteractive run:

```bash
MPLBACKEND=Agg venv/bin/python presnap/training/random_forest_trainer.py
```

The trainer does not currently save metrics automatically; this report and JSON record were captured separately. The additional majority baseline uses `DummyClassifier(strategy="most_frequent")` fitted on training labels and scored on validation labels.
