# Online score persistence — CSV schemas

Canonical schemas for `data/15_scores/online` latest and history files.

## Latest run per level

**Path:** `data/15_scores/online/latest/level{level}/latest_run_summary.csv`  
**Purpose:** Single row per level: the most recent archived run (by `created_at`) across all approaches.

| Column | Type | Description |
|--------|------|-------------|
| total_score | float | Net economic benefit (savings − fees) |
| total_savings | float | Sum of savings from correctly predicted core items |
| total_fees | float | Fixed fee × num_predictions |
| num_hits | int | Predicted (buyer, product) pairs matching ground truth |
| num_predictions | int | Number of predicted items |
| spend_capture_rate | float | Fraction of ground-truth spend captured |
| total_ground_spend | float | Total ground-truth core-demand spend |
| approach | string | Approach name (e.g. baseline, lgbm_two_stage) |
| run_id | string | Timestamp_sha[_dirty] |
| run_dir | string | Path to run folder under 15_scores/online/runs |
| commit_sha | string | Git commit at archive time |
| branch | string | Git branch at archive time |
| dirty | string | "true" or "false" |
| created_at | string | ISO8601 UTC (e.g. 2026-03-07T16:38:34Z) |

## Runs history (append-only)

**Path:** `data/15_scores/online/history/level{level}/runs_history.csv`  
**Purpose:** One row per archived run; append-only audit log.

| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Timestamp_sha[_dirty] |
| approach | string | Approach name |
| level | int | Level (1 or 2) |
| created_at | string | ISO8601 UTC |
| commit_sha | string | Git commit at archive time |
| branch | string | Git branch at archive time |
| dirty | string | "true" or "false" |
| run_dir | string | Path to run folder |
| total_score | float | From score_summary |
| total_savings | float | From score_summary |
| total_fees | float | From score_summary |
| num_hits | int | From score_summary |
| num_predictions | int | From score_summary |
| spend_capture_rate | float | From score_summary |
| total_ground_spend | float | From score_summary |

## Live submissions history (append-only)

**Path:** `data/15_scores/online/history/level{level}/submissions_live_history.csv`  
**Purpose:** One row per portal submission; append-only audit log.

| Column | Type | Description |
|--------|------|-------------|
| submission_id | string | Portal API submission id (UUID) |
| approach | string | Approach name |
| level | int | Level (1 or 2) |
| submitted_at_utc | string | ISO8601 UTC when we wrote the row |
| submission_path | string | Path to submitted CSV |
| total_score | float | Parsed from portal (or empty) |
| total_savings | float | Parsed from portal (or empty) |
| total_fees | float | Parsed from portal (or empty) |
| num_hits | int | Parsed from portal (or empty) |
| num_predictions | int | Parsed from portal (or empty) |
| spend_capture_rate | float | Parsed from portal (or empty) |
| total_ground_spend | float | Parsed from portal (or empty) |
