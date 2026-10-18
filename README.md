# Ethics of Artificial Intelligence -- Practical Assignment

## Development

Use Python 3.12 and the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

`pytest.ini` adds `src` to the import path. Tests require no custom
`PYTHONPATH`, exclusions, or graphical display. Use the virtual environment's
`pytest`; a system-wide installation may use a different Python without the
project dependencies. Coverage is optional: `pytest --cov=src`.

The environment can also be
created by id, like the original Mosaic ones (`import soccer` registers the ids):

```python
import soccer, gymnasium as gym
env = gym.make("EthicsSoccer-3v3-v0", render_mode="human")  # or EthicsSoccer-2v2-v0
```

Save a picture of the current state with `soccer.rendering.pitch.save_frame(env, 'pitch.png')`.

Run reproducible headless matches with:

```bash
python src/validate_matches.py --seeds 5 --output /tmp/soccer-validation.json
```

Source organization:

```text
src/
├── watch.py                 # checkpoint playback and automatic evaluation
├── mappo_train.py           # training workflow
├── validate_matches.py      # complete-match invariants and replay checks
├── enums.py
├── algorithms/mappo/        # networks, rollouts, PPO, checkpoints and devices
├── shared/                  # environment-independent utilities
└── soccer/
    ├── env.py               # match state and orchestration
    ├── config.py
    ├── entities/            # players and ball
    ├── rules/               # movement, ball rules, discipline and rewards
    ├── observation/         # full-state observations and network encoding
    ├── rendering/           # pitch drawing
    └── evaluation/          # event logging and fair-play statistics
```

Use `python src/watch.py --help`, `python src/mappo_train.py --help`, or
`python src/validate_matches.py --help` for the workflow options. Each workflow
has one implementation; no duplicate package scripts or demo are retained.

The ball and discipline classes supply rule methods directly to `SoccerEnv`;
they do not keep separate copies of match state or add delegation wrappers.
`SoccerEnvConfig` remains importable from `soccer.env` for existing callers.

### MAPPO training and watching

```bash
source .venv/bin/activate
python src/mappo_train.py --device auto --output checkpoints/run1
python src/watch.py checkpoints/run1/update_0010.pt --headless --device cpu
```

The default trains Team 0 against random. Add `--self-play` to train both
independent actor/critic pairs. Set `--team-size 2` for a separate 2v2 run.
Devices include `auto`, `cpu`, `cuda[:index]` and `mps`; accelerator execution
requires a compatible PyTorch build and hardware. See
[MAPPO integration](docs/MAPPO_INTEGRATION.md) for rollout, checkpoint and GPU details.

Playback automatically creates a unique folder under `evaluations/` containing
`events.jsonl`, `evaluation.json`, and `statistics.json`. See
[match logging and statistics](docs/MATCH_LOGGING.md).
