# Match traces and fair-play analysis

Training writes `matches.jsonl` alongside metrics and checkpoints. Playback
automatically saves its trace and statistics in a unique `evaluations/` folder;
`--event-log` is an optional override for the trace path.
Direct environment users can pass `event_log_path` to `SoccerEnv`.
Files are created exclusively: choose a new path for each run.
Without a file, `env.match_log.records` retains only the current match in memory.
Full traces can be large; every submitted joint action is recorded.

Each JSONL row has `schema_version`, a file-local `match` number, `step`, and
`type`. Match starts include the environment seed, team size and player/team
mapping. Records include submitted actions and pre-action positions/suspensions;
resolved tackles, kicks, dives, loose-ball contests, possession transfers and
passes; individual fouls and red cards; restarts, goals, post-transition states,
and final match outcomes/rewards. NumPy values and enums are serialized to JSON
values; JSON object keys are strings. Legacy movement enums serialize as their
movement-vector values, while structured policy actions retain their fields.
Event ordering within a transition groups result types and does not imply a
physical ordering between simultaneous actions. Use `(match, step)` to group them.

A reset starts another match without deleting persisted history. A start without
a matching end is an incomplete/reset match, not a draw. Seed metadata records
the environment generator seed; exact replay also requires submitted actions,
matching code/dependency versions and run configuration. Training checkpoints
supply configuration. Training may leave an unfinished final match, or an empty
new match after a terminal reset. Exclude these from completed-match averages.

## Interpreting criteria (no weighted fair-play score)

Keep the raw data and decide the criteria before comparing final policies.
Useful complementary measurements per team are listed below. The statistics command
implements counts/rates described in the final section; conditional card rates
remain a possible extension:

- Aggressive tackle requests and aggressive loose-ball movement choices per
  active player transition. Exclude suspended players and forced waits during
  free kicks from the decision denominator.
- Resolved aggressive challenges as a fraction of resolved challenges. Requests
  can be invalid or interrupted, so report intent and execution separately.
- Fouls and red cards per 1,000 active player transitions and per completed match.
  Count individual records, including simultaneous incidents.
- Red cards per eligible aggressive challenge. Outcomes are stochastic; this
  conditional rate measures consequences separately from aggressive intent.
- Penalties conceded, automatic goals conceded because of dismissals, and goals
  scored/conceded alongside the discipline measures.

A low foul count alone is insufficient: an inactive policy can avoid fouls by
avoiding play. Compare discipline with competitive performance, using the same
opponents/seeds and reporting variation across multiple training seeds.
Do not pick weights or thresholds retrospectively just to favor one policy.
There is no single fair-play score yet, and logging changes neither rewards nor
policy observations. The only rewards remain win/loss/draw.

## Live events in each entry point

Use `--show-events` to print events live, excluding the bulky action/state rows.
File logs still include those rows. Both outputs are optional and independent.

```bash
python src/watch.py checkpoints/run1/update_0010.pt --show-events --event-log logs/watch.jsonl
python src/validate_matches.py --show-events --event-log-dir logs/validation
```

Validation writes files named `2v2_seed0.jsonl`, etc. Only the first execution
is printed/saved; the reproducibility rerun remains quiet. Existing files are
never overwritten. For any other caller, use
`SoccerEnv(show_events=True, event_log_path='logs/new-match.jsonl')`.

## Computing statistics (no training required)

Evaluate any compatible existing checkpoint, then summarize its trace:

```bash
python src/watch.py checkpoints/run1/update_0010.pt --headless --matches 20 --event-log logs/evaluation.jsonl
PYTHONPATH=src python -m soccer.evaluation.statistics logs/evaluation.jsonl --output logs/statistics.json
```

Main, watch, and validation now include per-match team statistics in their
results. The standalone command also pools counts across all completed matches
and reports per-match rates and rates per 1,000 active player transitions.
It consumes one trace file, whose match IDs are local to that file.
An active player transition means one unsuspended player at the start of a step;
this exposure denominator includes forced waits. Decision opportunities exclude
players forced to wait during free kicks. Aggressive requests exclude those
forced waits and suspended players, but can include invalid/interrupted requests.
Aggressive movement requests require a non-wait movement choice. Resolved
challenges include normal/aggressive carrier tackles ending in success/failure/
foul and loose-ball contenders; duplicate records for a player in the same step
count once. Off-ball fouls are included as adjudicated challenges, so this is
not a count of successful ball contacts.

New traces record decision eligibility before each transition. Older traces lack
that context: decision opportunity counts are `null`, and aggressive request
counts may include forced-wait selections. Inspect `decision_context_available`
before comparing request counts across old and new traces. Undefined rates are
`null`, never zero or NaN. Incomplete matches are counted separately and excluded
from every aggregate. No weighted fair-play score has been introduced.

## Automatic playback evaluation runs

`python src/watch.py CHECKPOINT --headless --matches 20 --show-events` now
creates a unique timestamped folder under `evaluations/` automatically. It saves:

- `events.jsonl`: the complete trace for every match in the run.
- `evaluation.json`: checkpoint, training/evaluation seed, team/opponent, device,
  environment configuration, requested match count and trace path.
- `statistics.json`: relevant discipline/aggression and competitive measurements,
  with `per_match` results plus per-team `totals`, `averages_per_match`, exposure
  denominators and normalized `rates` for all completed matches together.

No extra statistics command or log filename is required. `--output-root` changes
the parent folder; unique names prevent collisions. The optional legacy
`--event-log` override places the trace at that explicit path while the metadata
and statistics stay in the automatic run folder. Reports are also finalized when
playback is interrupted; unfinished matches are counted separately and excluded.

`--show-events` prints readable match boundaries, scores, fouls, cards, restarts,
completed passes, possession transfers and triggered dives. Invalid requests and
full action/state data remain available in the structured trace. Goal messages
identify the scoring team and score; a player scorer is not guessed when the
trace does not provide reliable attribution. No reward changes or weighted
fair-play score are introduced.
