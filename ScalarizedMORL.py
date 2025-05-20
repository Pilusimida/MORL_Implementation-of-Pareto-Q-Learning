from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym

# Hyperparameters
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.999
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9998

# 将状态转换为可哈希的元组
def hash_state(state):
    return tuple(state.round(2))  # 四舍五入减少状态空间

# Environment setup
env = mo_gym.make('deep-sea-treasure-v0')
action_dim = env.action_space.n

# Weight combinations (10 linear weightings between [0,1] and [1,0])

weights = np.arange(0.1, 1, 0.1)
weights = [(w, 1 - w) for w in weights]

# Results storage
pareto_front = []

for weight in weights:
    print(f"\nTraining with weight: {weight}")

    # Initialize Q-table (state x action)
    Q = defaultdict(lambda: [[0.0, 0.0] for _ in range(action_dim)])
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        # print(episode)
        state, _ = env.reset()
        state_hashed = hash_state(state)
        done = False
        # Epsilon decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                # Scalarize Q-values using current weight
                scalar_q = [weight[0]*Q[state_hashed][a][0] + weight[1]*Q[state_hashed][a][1] for a in range(action_dim)]
                action = np.argmax(scalar_q)  # Exploit

            # Take action
            next_state, reward, done, _, _ = env.step(action)

            # Scalarized target
            target = [0, 0]
            if done:
                target = reward
            else:
                next_state_hashed = hash_state(next_state)
                next_scalar_q = [weight[0]*Q[next_state_hashed][a][0] + weight[1]*Q[next_state_hashed][a][1] for a in range(action_dim)]
                best_action = np.argmax(next_scalar_q)
                target[0] = reward[0] + GAMMA * Q[next_state_hashed][best_action][0]
                target[1] = reward[1] + GAMMA * Q[next_state_hashed][best_action][1]

            # Update Q-values for both objectives
            Q[state_hashed][action][0] += ALPHA * (target[0] - Q[state_hashed][action][0])
            Q[state_hashed][action][1] += ALPHA * (target[1] - Q[state_hashed][action][1])

            state = next_state
            state_hashed = hash_state(state)



    # Extract Pareto-optimal solutions
    # Find terminal state
    state, _ = env.reset()
    done = False
    steps = 0
    while not done:
        # Greedy action selection
        # Scalarize Q-values using current weight
        state = hash_state(state)
        scalar_q = [weight[0] * Q[state][a][0] + weight[1] * Q[state][a][1] for a in range(action_dim)]
        action = np.argmax(scalar_q)  # Exploit

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        steps -= 1
        # Update state
        state = next_state
    pareto_front.append([reward[0], steps])  # (treasure, time_penalty)

# Plot results
treasures, times = zip(*pareto_front)

plt.figure(figsize=(10, 6))
plt.scatter(times, treasures, c='red', marker='o')
plt.xlabel('Time Penalty (Steps)', fontsize=20)
plt.ylabel('Treasure Value', fontsize=20)
plt.title('Deep Sea Treasure: Scalarized Q-learning Pareto Set', fontsize=20)
plt.grid(True)
plt.savefig('Result_SQL.pdf', dpi=300, bbox_inches='tight')
plt.show()
