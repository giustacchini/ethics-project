import gymnasium as gym
import mosaic_multigrid.envs

env = gym.make("MosaicMultiGrid-S-3v3-IndAgObs-v1", render_mode="human")
obs, info = env.reset(seed=0)

total = {i: 0.0 for i in obs}
step = 0
done = False

while not done:
    # One random action per agent, all sent simultaneously
    actions = env.action_space.sample()
    obs, rewards, terms, truncs, infos = env.step(actions)

    for i, r in rewards.items():
        total[i] += r
    step += 1

    # Write out when something happens
    for key in ("goal_scored_by", "pass_completed", "steal_completed"):
        if key in infos[0]:
            print(f"Step {step}: {key} -> {infos[0][key]}")

    done = all(terms.values()) or all(truncs.values())

print(f"\nMatch finished after {step} steps")
print("Score:", env.unwrapped.team_scores)
print("Total reward per agent:", {i: round(r, 2) for i, r in total.items()})
env.close()