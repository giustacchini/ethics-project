# Soccer MAPPO integration

## Implemented interface batch

The intended model has four networks total: one actor shared by each team's
players and one centralized critic per team. `create_teams` in
`algorithms/mappo/teams.py` constructs independent networks and Adam
optimizers for Team 0 and Team 1. Training team sizes independently remains
the experimental baseline.

`SoccerEnv` and its match-outcome-only rewards are unchanged. A random
opponent still samples its environment action space; policy actions simply
replace the trained team's entries in the joint action dictionary.

## Observation encoding

`soccer.observation.encoding.encode_team` returns sorted player IDs, a float32 actor
observation matrix, one centralized team-state vector, and an active-player
mask. Actor vectors put self first, then teammates and opponents. Critic
vectors use a canonical team-first ordering independent of the observer.
Absolute player IDs and the team's absolute side remain encoded, so action
target IDs and attack direction are unambiguous.

Fields cover player positions/teams, original/current roles, suspensions,
carrier/kicker/free-kick identity, ball presence/state/position/speed and
last possession, both scores, time, match/goal state and complete remaining
trajectory. Trajectories are padded with validity flags to the maximum
straight-line length on the pitch. Missing positions retain (-1,-1).
The encoding version is explicit; old 23-value checkpoints are incompatible.

## Structured actions

The actor has categorical heads for type, movement, target x/y, strength,
player target and contest intensity. The relevant heads are:

| Action | Heads beyond action type |
|---|---|
| Wait | None |
| Move | Movement, contest intensity |
| Kick | Target x/y, strength |
| Tackle / aggressive tackle | Player target |
| Dive | Target x/y |

`SoccerActionDistribution.log_prob` sums only relevant head terms.
`entropy(actions)` likewise uses conditional head entropy for the sampled
or evaluated action type. `actions_to_env` decodes actor batches directly
into the existing environment dictionaries.

An inactive-player mask forces deterministic waiting with zero action
log-probability/entropy. `masked_actor_mean` averages loss terms over active
players only (all-inactive batches produce zero); training must use this
mask rather than treating inactivity as episode termination. No sporting
behavior is prescribed by action masking: off-ball and aggressive tackles
remain available, and the environment retains game-rule validation.

## Training and evaluation

The rollout collector, per-team buffers, GAE, clipped PPO update, trainer and
watcher now use SoccerEnv directly. There is no hidden environment wrapper
or automatic reset inside SoccerEnv: the collector resets after complete
matches and retains episode accounting across rollout chunks.

```bash
source .venv/bin/activate
# One team learns against the unchanged random-policy baseline.
python src/mappo_train.py --team-size 3 --device auto --output checkpoints/random_seed0
# Both independent teams learn from the same joint transitions.
python src/mappo_train.py --team-size 3 --self-play --device auto --output checkpoints/selfplay_seed0
# Save every five updates; the final update is always saved.
python src/mappo_train.py --updates 20 --checkpoint-interval 5 --output checkpoints/interval_seed0
# Resume for ten additional updates from an existing checkpoint.
python src/mappo_train.py --resume checkpoints/interval_seed0/update_0020.pt --updates 10
# Watch the last checkpoint (add --self-play for a self-play checkpoint).
python src/watch.py checkpoints/random_seed0/update_0010.pt --device auto
# Evaluate without opening a window.
python src/watch.py checkpoints/random_seed0/update_0010.pt --headless --matches 3 --device cpu
```

Use distinct output directories/seeds for independent experiments. Existing
run outputs/checkpoints are not overwritten for new runs. Resume defaults to
the checkpoint's directory and continues checkpoint numbering; use `--output`
to select another directory. Defaults are startup settings,
not validated training optima; rewards are sparse and smoke tests do not
establish learning quality. Train 2v2 and 3v3 independently using --team-size.
--training-team chooses 0 or 1 against random; --self-play trains both.

Buffers store actions/likelihoods/masks per player, but rewards, critic states,
values and returns once per team transition. GAE masks the next match at
both natural termination and the match time limit, since both produce real
terminal outcome rewards here. PPO shares each team advantage across its
active decision-making players and trains the critic once per transition.
Suspended players and players forced to wait during a free kick are excluded
from actor updates. All-inactive actor batches are skipped, while centralized
critic training continues. Single-sample advantage normalization is finite.

Each update writes metrics.jsonl records and a versioned checkpoint containing
both network pairs, optimizer states, training/environment/network configuration,
encoding version, seed and update number. Watching checks schema compatibility
and refuses untrained team selections. Old flat-action checkpoints are not
compatible. Checkpoints support evaluation and training resume. Resume starts
a fresh environment rollout from the saved network and optimizer state; exact
mid-match continuation is not implemented (collector/RNG/rollout state is not
saved). `--updates` means additional updates when resuming.
`--checkpoint-interval` controls periodic saves, and the final update is
always saved.

## GPU support

--device auto chooses available CUDA first, then Apple MPS, then CPU.
Explicit --device cpu, --device cuda, --device cuda:1 or --device mps requests
are checked and produce clear errors when unavailable. Networks, inputs,
rollouts, GAE, minibatch indices and optimizer updates use the selected
device. Environment simulation/rendering stays on CPU; actions cross back
to the environment as ordinary dictionaries. Checkpoints load with explicit
map_location so GPU checkpoints can be evaluated on CPU.

The local environment has CPU-only PyTorch, so CPU execution and device
selection/error handling were tested. Real accelerator execution was not
available to test. Install the build appropriate for your hardware using
the [official PyTorch selector](https://pytorch.org/get-started/locally/)
before requesting CUDA. CPU environment simulation remains a possible
throughput bottleneck; GPU selection alone is not a speedup guarantee.

## Dependencies and checks

PyTorch is now declared in requirements.txt. CPU-only installation can be
selected before installing the remaining requirements:

```bash
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
pytest
```

The interface tests cover complete-state encoding, 2v2/3v3 environment
stepping against random, head-specific likelihoods/gradients, suspension
masking and independent team parameters. The full suite now passes 323 tests, including rollout boundaries, masking,
finite optimizer updates, independent teams, device handling and checkpoint
round trips. A two-update CLI run and headless checkpoint evaluation also
passed on CPU.
