"""Soccer environment components.

Importing this package registers the environment ids with gymnasium, so the
env can be created like the original Mosaic one::

    import soccer, gymnasium as gym
    env = gym.make("EthicsSoccer-3v3-v0", render_mode="human")
"""

import gymnasium as gym

# Entry points are strings, so soccer.env is only imported when gym.make is
# called (avoids circular imports). The env uses Mosaic's multi-agent dict API
# (not the single-agent gym API), so gym's single-agent checker is disabled.
for _id, _entry in (
    ("EthicsSoccer-3v3-v0", "soccer.env:make_3v3"),
    ("EthicsSoccer-2v2-v0", "soccer.env:make_2v2"),
):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_entry, disable_env_checker=True)
