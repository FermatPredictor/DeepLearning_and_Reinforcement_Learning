# -*- coding: utf-8 -*-
# q_learning.py
# https://geektutu.com
"""
這邊是純樸的 Q-learning算法，
有用神經網路版的DQN方法跑起來有點過慢
"""


from collections import defaultdict
import gym  # 0.12.5
import numpy as np
import pickle

# 默认将Action 0,1,2的价值初始化为0
Q = defaultdict(lambda: [0, 0, 0])

env = gym.make('MountainCar-v0')



def transform_state(state):
    """将 position, velocity 通过线性转换映射到 [0, 40] 范围内"""
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)

# print(transform_state([-1.0, 0.01])) # eg: (4, 22)


lr, factor = 0.7, 0.95
episodes = 15000  # 训练10000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = transform_state(env.reset())
    score = 0
    while True:
        a = np.argmax(Q[s])
        # 训练刚开始，多一点随机性，以便有更多的状态
        if np.random.random() > i * 3 / episodes:
            a = np.random.choice([0, 1, 2])
        # 执行动作
        next_s, reward, done, _ = env.step(a)
        next_s = transform_state(next_s)
        # 根据上面的公式更新Q-Table
        Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 -160 时，停止并保存模型
    if np.mean(score_list[-10:]) > -130:
        with open('MountainCar-v0-q-learning.pickle', 'wb') as f:
            pickle.dump(dict(Q), f)
            print('model saved')
            break
env.close()

import matplotlib.pyplot as plt

plt.plot(score_list, color='green')
plt.show()
