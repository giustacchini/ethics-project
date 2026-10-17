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

Run the visual demo with `python src/main.py --seed 0`. The env can also be
created by id, like the original Mosaic ones (`import soccer` registers the ids):

```python
import soccer, gymnasium as gym
env = gym.make("EthicsSoccer-3v3-v0", render_mode="human")  # or EthicsSoccer-2v2-v0
```

Save a picture of the current state with `soccer.rendering.save_frame(env, 'pitch.png')`.

Run reproducible headless matches with:

```bash
PYTHONPATH=src python -m soccer.validate_matches --seeds 5 --output /tmp/soccer-validation.json
```

Source organization:

- `src/enums.py`: action and event enums.
- `src/shared/`: domain-independent movement stages, grid helpers, events, rewards and the observation interface.
- `src/soccer/config.py`: match configuration.
- `src/soccer/env.py`: match state and transition orchestration.
- `src/soccer/observations.py`: full structured observations and their spaces.
- `src/soccer/ball_rules.py`: kicks, possession and dives.
- `src/soccer/discipline.py`: tackles, suspensions and restarts.
- `src/soccer/agent.py`, `ball.py`: domain objects (players and ball).
- `src/soccer/rendering.py`: pitch drawing in the original Mosaic soccer style
  (striped grass, markings, net goals, red ball). Goalkeepers are squares and
  field players are triangles. It only reads the env; it never changes game state.
- `src/soccer/validate_matches.py`: complete-match invariants and replay checks.
- `tests/unit/shared/` and `tests/unit/soccer/`: tests organized by domain.

The ball and discipline classes supply rule methods directly to `SoccerEnv`;
they do not keep separate copies of match state or add delegation wrappers.
`SoccerEnvConfig` remains importable from `soccer.env` for existing callers.

### MAPPO training and watching

```bash
source .venv/bin/activate
PYTHONPATH=src python -m algorithms.mappo.train --device auto --output checkpoints/run1
PYTHONPATH=src python -m soccer.watch checkpoints/run1/update_0010.pt --headless --device cpu
```

The default trains Team 0 against random. Add `--self-play` to train both
independent actor/critic pairs. Set `--team-size 2` for a separate 2v2 run.
Devices include `auto`, `cpu`, `cuda[:index]` and `mps`; accelerator execution
requires a compatible PyTorch build and hardware. See
[MAPPO integration](docs/MAPPO_INTEGRATION.md) for rollout, checkpoint and GPU details.
