'''
画假图
'''

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.utils import ornstein_uhlenbeck_process
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde

# 设置Matplotlib使用中文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']


# Parameters
# np.random.seed(2)
# random.seed(2)
theta = 0.9  # Speed of mean reversion
mu = 0.0     # Long-term mean
sigma = 0.1  # Volatility
x0 = 1.0     # Initial value
dt = 0.01    # Time increment
n_steps = 500  # Number of steps

# 生成线性递减的序列
mu = np.linspace(1.29, 0.78, n_steps)


ou_process1 = ornstein_uhlenbeck_process(theta, mu, sigma, x0, dt, n_steps)
ou_process2 = ornstein_uhlenbeck_process(theta, mu, sigma, x0, dt, n_steps)
ou_process3 = ornstein_uhlenbeck_process(theta, mu, sigma, x0, dt, n_steps)
x = np.linspace(0, 1, n_steps)

# processed_ou = []
# for i in range(len(ou_process)):
#     if i == 0:
#         processed_ou.append(ou_process[i])
#     else:
#         processed_ou.append(ou_process[i] if ou_process[i] <= processed_ou[i-1] else processed_ou[i-1])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

color0 = [53/255, 18/255, 63/255]
color1 = [47/255, 163/255, 132/255]
color2 = [156/255, 212/255, 118/255]
colors = [color0, color1, color2]

for i, y in enumerate([ou_process1, ou_process2, ou_process3]):
    verts = [[(0, i/2, 0)] + [(x[j], i/2, y[j]) for j in range(n_steps)] + [(x[-1], i/2, 0)]]
    poly = Poly3DCollection(verts, facecolors=colors[i], edgecolors=colors[i], alpha=0.3, lw=2)
    ax.add_collection3d(poly)
ax.legend(['OU1', 'OU2', 'OU3'])
# ax.set_yticklabels(['ou1', 'ou2', 'ou3'])
# ax.tick_params(axis='x', which='major', pad=15)
# plt.plot(processed_ou)
plt.grid()
# plt.xlabel('进化代数', fontsize=18)
# plt.ylabel('适应度', fontsize=18)
# plt.tick_params(axis='x', labelsize=18)  # 设置 x 轴刻度字号
# plt.tick_params(axis='y', labelsize=18)  # 设置 y 轴刻度字号

ax.view_init(elev=21, azim=41)
plt.show()