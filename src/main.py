from soccer.env import SoccerEnv

env = SoccerEnv(render_mode="human")
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

    done = all(terms.values()) or all(truncs.values())

print(f"\nMatch finished after {step} steps")
print("Total reward per agent:", {i: round(r, 2) for i, r in total.items()})
env.close()
