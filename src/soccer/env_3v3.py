import gymnasium as gym
import mosaic_multigrid.envs

env = gym.make("MosaicMultiGrid-S-3v3-IndAgObs-v1", render_mode="human")

# ESTABLISH INITIAL POSITIONS AFTER THE ENVIRONMENT IS RESET
obs, info = env.reset(seed=0)

# 1. Initial positions
initial_positions = [
    (1, 5),   # Goalkeeper left
    (4, 3),   # Player left
    (4, 7),   # Player left
    (14, 5),  # Goalkeeper right
    (11, 3),  # Player right
    (11, 7)   # Player right
]

# 2. Set the initial positions and directions of the agents
for i, pos in enumerate(initial_positions):
    env.unwrapped.agents[i].pos = pos
    # Team 1 (0, 1, 2) looks to the right (0)
    # Team 2 (3, 4, 5) looks to the left (2)
    env.unwrapped.agents[i].dir = 0 if i < 3 else 2

# 3. Place the ball in the center of the field
ball = None
vieja_x, vieja_y = -1, -1

for x in range(env.unwrapped.width):
    for y in range(env.unwrapped.height):
        obj = env.unwrapped.grid.get(x, y)
        if obj is not None and obj.type == 'ball':
            ball = obj
            vieja_x, vieja_y = x, y
            break
    if ball is not None:
        break

if ball is not None:
    env.unwrapped.grid.set(vieja_x, vieja_y, None)  # Move the ball from its old position
    ball.pos = (8, 5)
    env.unwrapped.grid.set(8, 5, ball)            # Place it in the exact center

# 4. Make a "blank step" to update all cameras
obs, rewards, terms, truncs, infos = env.step({i: 0 for i in range(6)})


# Print the initial positions of all players
print("Jugadores en el campo:", list(obs.keys()))

from mosaic_multigrid.core.actions import Action
print("Controles exactos:", {a.name: a.value for a in Action})

total = {i: 0.0 for i in obs}
step = 0
done = False

while not done: 
    
    # RESTRICT GOALKEEPERS' MOVEMENTS TO THEIR DESIGNATED AREA
    # Sample random actions for all agents
    actions = env.action_space.sample()

    # 1. Create an anti-crash shield to avoid hitting the walls (x=0 and x=15)
    for i in range(6):
        if actions[i] in [3, 4, 5, 6]:
            nx, ny = env.unwrapped.agents[i].front_pos
            if nx <= 0 or nx >= 15 or ny <= 0 or ny >= 10:
                actions[i] = 0

    # 2. Goalkeepers' movement restrictions
    goalkeeper_limits = {
        0: (1, 2),   # Goalkeeper left
        3: (13, 14)  # Goalkeeper right
    }
    
    for gk_id, (min_x, max_x) in goalkeeper_limits.items():
        if actions[gk_id] == 3: # Move forward
            nx, ny = env.unwrapped.agents[gk_id].front_pos
            if not (min_x <= nx <= max_x) or not (2 <= ny <= 8):
                actions[gk_id] = 0

    # 3. Execute the step with the modified actions
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    

    for i, r in rewards.items():
        total[i] += r
    step += 1

    # Write out when something happens
    for key in ("goal_scored_by", "pass_completed", "steal_completed"):
        if key in infos[0]:
            print(f"Step {step}: {key} -> {infos[0][key]}")
            
            # After a goal is scored, reset the positions of the players and the ball
            if key == "goal_scored_by":
                
                # 1. Reposition the players
                for i, pos in enumerate(initial_positions):
                    env.unwrapped.agents[i].pos = pos
                    env.unwrapped.agents[i].dir = 0 if i < 3 else 2
                
                # 2. Reposition the ball
                ball = None
                vieja_x, vieja_y = -1, -1
                for x in range(env.unwrapped.width):
                    for y in range(env.unwrapped.height):
                        obj = env.unwrapped.grid.get(x, y)
                        if obj is not None and obj.type == 'ball':
                            ball = obj
                            vieja_x, vieja_y = x, y
                            break
                    if ball is not None:
                        break
                
                if ball is not None:
                    env.unwrapped.grid.set(vieja_x, vieja_y, None)
                    ball.pos = (8, 5)
                    env.unwrapped.grid.set(8, 5, ball)

    done = all(terms.values()) or all(truncs.values())
    

print(f"\nMatch finished after {step} steps")
print("Score:", env.unwrapped.team_scores)
print("Total reward per agent:", {i: round(r, 2) for i, r in total.items()})
env.close()