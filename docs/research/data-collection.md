# Data collection evidence

Updated October 8, 2026.

## Source and local acquisition

- Available release: https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data
- Original tracking collection: https://edge-operations.nfl.com/game-operations-logistics/technology/performance-tracking-data-next-gen-stats
- NFL organizer: https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl
- Initial Kaggle source, now unavailable: https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data

The regional publisher attributes tracking to NFL Next Gen Stats and scouting/coverage labels to PFF. Detailed PFF annotation procedures and a standalone data license have not been established. The 2021 regional release must not be described as the 2025 Kaggle release.

All 126 files were downloaded into ignored `data/regional-2021/`. The local manifest records the source commit, byte sizes and SHA-256 hashes. Data records and source examples remain local except the compact summaries included in the proposal.

## Verified initial audit

- 122 games, 2021 season weeks 1–8.
- 8,557 distinct play records with 12 observed PFF coverage classes.
- 122 tracking CSVs with 8,314,178 rows.
- 8,522 labeled plays have a complete strictly pre-snap snapshot with 22 distinct players, 11 per team, and nonmissing coordinates.
- 35 plays fail these initial snapshot checks. This does not establish overall annotation accuracy; further modeling exclusions may reduce the count.

Run `uv run --with pandas python scripts/audit_regional_data.py` from the repository root to regenerate ignored `audit.json`, including three source examples with different labels. The script validates completeness before selecting the last qualifying pre-snap frame. The proposal examples report the defensive coordinate ranges calculated from those actual frames, alongside down/distance and exact source coverage labels.

Schema changes: tracking uses `team`, not `club`, and has no `frameType`. Use frame IDs strictly below `ball_snap`. The related coverage annotation is `pff_passCoverageType`, not `pff_manZone`. Neither target-related annotation belongs in model inputs.

## Rubric coverage and limits

The data section now addresses sources, collection origin, access, annotations, available metadata, subset rationale, preprocessing, measured size, and three labeled examples (items 5–6). Applicable sharing terms remain a verification item; public accessibility is not an unrestricted license. Class frequencies and preprocessing should be reviewed when fixing the task's class taxonomy and rare-class handling.

The task-definition, solution/evaluation, and contract placeholders belong to the remaining team contributions. Final submission still needs the consulted TA at the top, the page/format requirements, and separate proposal and signed contract PDFs. The native editor compiles the current source; the checked-in PDF is the upstream artifact and must be exported again when the team finalizes the document.

Relevant prior work for the solution owner: https://www.nfl.com/news/next-gen-stats-new-advanced-metrics-you-need-to-know-for-the-2025-nfl-season describes an existing pre-snap coverage-disguise model. Avoid an unsupported claim that this task is entirely new.
