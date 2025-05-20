
import mo_gymnasium as mo_gym
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict
np.random.seed(0)
# 初始化环境
env = mo_gym.make('deep-sea-treasure-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
obj_dim = env.unwrapped.reward_space.shape[0]  # 目标维度（时间 vs 宝藏价值）

# 超参数
EPISODES = 2000
ALPHA = 0.1  # 学习率
# GAMMA = 0.99  # 折扣因子
GAMMA = 0.999  # 折扣因子
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9998

# 将状态转换为可哈希的元组
def hash_state(state):
    return tuple(state.round(2))  # 四舍五入减少状态空间

# Pareto Q-Learning 核心数据结构
Q_ND = defaultdict(lambda: [[] for _ in range(action_dim)])  # Q(s,a) 是多目标向量，某个状态下每一个action对应的q值向量集合
V_ND = defaultdict(list)  # 某一个状态对应的所有帕累托非支配的值向量
R_avg = defaultdict(lambda: [[0 for i in range(obj_dim)] for j in range(action_dim)])  # 存储平均即时奖励
access_counter = defaultdict(lambda: [0 for j in range(action_dim)])  # 存储访问次数
# 判断是否支配
def dominates(reward1, reward2):
    """检查 reward1 是否 Pareto 支配 reward2"""
    return np.all(reward1 >= reward2) and np.any(reward1 > reward2)


def calculate_HV(points, reference_point):
    # 按第一目标从小到大排序（即原始最大目标从大到小）
    front = sorted(points, key=lambda x: x[0])

    hv = 0.0
    for i in range(len(front)):
        x = front[i][0]
        y = front[i][1]
        if i == 0:
            width = x - reference_point[0]
        else:
            width = x - front[i-1][0]
        height = y - reference_point[1]
        hv += width * height
    return hv

def get_non_dominated(points):
    """
    获取多目标最大化问题中的所有非支配解（Pareto 前沿）

    参数:
    - points: List[List[float]]，每个子列表是一个目标向量（目标越大越好）

    返回:
    - List[List[float]]，所有非支配点
    """
    points = np.unique(np.array(points), axis=0)
    num_points = len(points)
    is_dominated = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                # 如果 point[j] 支配 point[i]（最大化问题）
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_dominated[i] = True
                    break

    return points[~is_dominated].tolist()


def HV_action_selection(state, epsilon, reference_point):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        hv_list = []
        state_key = hash_state(state)
        for action in range(action_dim):
            Q = Q_ND[state_key][action]
            hv = calculate_HV(Q, reference_point)
            hv_list.append(hv)
        return np.argmax(hv_list)



# points = [[0, 2], [1, 1], [2, 0], [0.5, 0.5]]
# reference_point = [0, 0]
# hv = calculate_HV(points, reference_point)
# print(hv)
# ND = get_non_dominated(points)
# print(ND)

# 主训练循环
reference_point = [0, -25]
episode_rewards = []
hv_record = []
episode_checkpoints = []

for episode in range(EPISODES):
    # print(f"Episode: {episode}")
    state, _ = env.reset()
    total_reward = np.zeros(obj_dim)
    done = False
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
    steps = 0
    while not done and steps < 200: # Not reaching terminal state
        steps += 1
        # Sample action using epsilon_greedy algorithm
        action = HV_action_selection(state, epsilon, reference_point)

        # Execute given action, observe next_state, reward and done
        next_state, reward, done, _, _ = env.step(action)

        # 更新 Q 值（多目标贝尔曼方程）
        state_key = hash_state(state)
        next_state_key = hash_state(next_state)

        # Update Average Reward
        # print(access_counter[state_key][action])
        access_counter[state_key][action] = access_counter[state_key][action] + 1
        previous_R = R_avg[state_key][action]
        R_avg[state_key][action] = previous_R + (np.array(reward) - np.array(previous_R)) / access_counter[state_key][
            action]

        # Update Q_ND(state, action)
        if done:
            Q_ND[state_key][action] = [list(reward)]
        else:
            waiting_Q = []
            for a in range(action_dim):
                waiting_Q = waiting_Q + Q_ND[next_state_key][a]
            ND_Q = get_non_dominated(waiting_Q)
            new_Q_list = []
            for q_vector in ND_Q:
                new_q = np.array(R_avg[state_key][action]) + GAMMA * np.array(q_vector)
                new_Q_list.append(new_q.tolist())
            Q_ND[state_key][action] = new_Q_list

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        waiting_Q = []
        state = np.array([0, 0])
        state_key = hash_state(state)
        for a in range(action_dim):
            for e in Q_ND[state_key][a]:
                waiting_Q.append(e)
        HV = calculate_HV(get_non_dominated(waiting_Q), reference_point)
        hv_record.append(HV)
        episode_checkpoints.append(episode)
        print(f"Episode {episode}, HV: {HV}, Epsilon: {epsilon:.2f}")

print("Q vectors in state (0, 0)")
action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
for action in range(action_dim):
    print(f"Action: {action_map[action]}")
    print(Q_ND[(0, 0)][action])


# 绘制 HV 曲线
plt.figure(figsize=(8, 5))
plt.plot(episode_checkpoints, hv_record, marker='o', linestyle='-', label="PQL")
max_HV = [401.8 for _ in range(len(episode_checkpoints))]
plt.plot(episode_checkpoints, max_HV, label="PF")
plt.title("Hypervolume (HV) over Training Episodes", fontsize=20)
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Hypervolume", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig('Training_Process_PQL.pdf', dpi=300, bbox_inches='tight')
plt.show()

