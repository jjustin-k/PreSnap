# Pre-snap coverage dataset

`data/presnap/` contains the subset used to predict defensive coverage from pre-snap positions. It retains 8,522 plays (187,484 player rows), excludes 35 incomplete plays, and occupies about 14 MB. The full regional release stays local in ignored `data/regional-2021/`.

Source: [NFL Big Data Bowl regional-event release](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data), revision `85da22eeff2f1d5be106faa9dfe06a1205f2defd`, covering weeks 1–8 of the 2021 season. This is separate from Big Data Bowl 2025.

## Included files

- `tracking.csv`: one complete snapshot per included play, containing all 22 players. Select defenders with `isDefense == 1`; retain offensive positions for alignment features. Coordinates and directions are source values, without field normalization.
- `plays.csv`: pre-snap game context and alignment fields. Join using `gameId` and `playId`.
- `labels.csv`: `pff_passCoverage` and `pff_passCoverageType` targets, kept separate from predictors.
- `excluded_plays.csv`: omitted play IDs and exclusion reasons.
- `summary.json`: retained counts, class distribution, source revision, and output CSV hashes.
- `source_manifest.json`: original source file sizes and hashes for reproducibility; it does not imply every source file is included here.

For each labeled play, select the latest frame strictly before the first `ball_snap` event that has valid x/y coordinates, 22 distinct player IDs, and exactly 11 players from each of the possession and defensive teams. Plays without such a frame are excluded. No post-snap tracking, play descriptions, outcomes, penalties, or player-level scouting records are included.

Coverage labels describe the executed coverage, not an independently annotated pre-snap appearance. Neither coverage column belongs in model inputs. Split training and evaluation by `gameId` to avoid sharing plays from the same game across splits. Normalize coordinates by `playDirection` and line of scrimmage during feature preparation. Missing context values and rare coverage classes still require decisions during modeling.

## Reproduce

From the repository root:

```sh
uv run --with pandas python scripts/prepare_presnap_data.py --download
```

This downloads missing raw plays/tracking files from the pinned source revision, verifies their hashes, and rebuilds the subset. Omit `--download` when raw files already exist. Generated CSVs are sorted by play and player ID. The summary records exact retained and excluded counts.
