# Ingestion and alignment features

Run from the repository root:

```bash
python3 presnap/pipeline/ingest.py
```

The pipeline joins `data/presnap/plays.csv`, `labels.csv`, and `tracking.csv` by both game and play ID, producing one cleaned row per retained play. The current output contains 8,515 rows and 56 columns, including 37 alignment features. Seven source plays are skipped by existing context/personnel validation.

## Coordinate conventions

Depth is measured in yards from the line of scrimmage, positive in the offense's attacking direction. `absoluteYardlineNumber` supplies the line coordinate; missing values use possession side and yardline number. Leftward plays are rotated 180 degrees. Lateral coordinates are centered on the offense's median lateral position; positive means left from the attacking offense's perspective. This formation-center proxy is not the ball position: the snapshot excludes the football.

Each snapshot must contain 22 distinct players, 11 per team, a single frame strictly before a consistent snap frame, consistent direction/team membership, and finite coordinates. Invalid snapshots raise ValueError and are reported as skipped by the CLI.

## Features

- Deepest and second-deepest defender depths, their lateral spacing, and total defensive width.
- Defender counts strictly beyond 5, 10, and 15 yards downfield.
- Offensive width; counts left/right of the formation center; players more than 1 yard behind the line; players more than 5 yards laterally from formation center.
- Mean/min/max Euclidean distance from those wide offensive players to their nearest defender. If there are no wide players, these values are zero and `offense_wide_count` is zero.
- Eleven defender depth/lateral coordinate pairs, ordered deepest first with lateral position as the tie-breaker. Input CSV row order and player IDs do not determine slot order.

These are geometry-based descriptions, not inferred safety, cornerback, or receiver role labels. Existing personnel parsing and coverage targets are unchanged. The trainer automatically consumes the additional numeric columns. Existing benchmark reports refer to the previous dataset; retrain to evaluate alignment features on the same game partitions.

## Checks

```bash
python3 -m unittest discover -s tests -p 'test_ingest_alignment.py'
```

Tests cover geometry, column schema, row-order invariance, direction normalization, missing-line fallback, invalid snapshot rejection, and formations without wide players.
