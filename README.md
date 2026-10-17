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

Run the visual demo with `python src/main.py --seed 0`. Run reproducible
headless matches with:

```bash
PYTHONPATH=src python -m soccer.validate_matches --seeds 5 --output /tmp/soccer-validation.json
```

Source organization:

- `src/enums.py`: action and event enums.
- `src/shared/`: domain-independent movement stages, grids, events and rewards.
- `src/soccer/config.py`: match configuration.
- `src/soccer/env.py`: match state and transition orchestration.
- `src/soccer/observations.py`: full structured observations and their spaces.
- `src/soccer/ball_rules.py`: kicks, possession and dives.
- `src/soccer/discipline.py`: tackles, suspensions and restarts.
- `src/soccer/agent.py`, `ball.py`, `grid.py`: domain objects and rendering.
- `src/soccer/validate_matches.py`: complete-match invariants and replay checks.
- `tests/unit/shared/` and `tests/unit/soccer/`: tests organized by domain.

The ball and discipline classes supply rule methods directly to `SoccerEnv`;
they do not keep separate copies of match state or add delegation wrappers.
`SoccerEnvConfig` remains importable from `soccer.env` for existing callers.
