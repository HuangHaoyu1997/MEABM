import random
import numpy as np
import matplotlib.pyplot as plt
from src.srn import NECN, encoding





# Parameters
np.random.seed(123)
random.seed(123)
theta = 0.7  # Speed of mean reversion
mu = 0.0     # Long-term mean
sigma = 0.3  # Volatility
x0 = 1.0     # Initial value
dt = 0.01    # Time increment
n_steps = 1000  # Number of steps

# Simulate the process
ou_process = ornstein_uhlenbeck_process(theta, mu, sigma, x0, dt, n_steps)
ou_process1 = ornstein_uhlenbeck_process(ou_process, mu, sigma, x0, dt, n_steps) # OU套OU

series = (ou_process-np.min(ou_process)+1)*100
series = [series[i*10] for i in range(n_steps//10)]

plt.subplot(5, 1, 1)
plt.plot([i for i in range(100)], series)  # np.linspace(0, n_steps * dt, n_steps)
plt.grid(True)
plt.ylabel('GDP')

ConceptNeurons = [
        NECN(v_th=5),
        NECN(v_th=10),
        NECN(v_th=20),
        NECN(v_th=50),
    ]
yts = []
for xt in series:
    yt, ConceptNeurons = encoding(xt, ConceptNeurons)
    yts.append(yt)
yts = np.array(yts) # (200, 4)
print('yts.shape: ', yts.shape)

plt.subplot(5, 1, 2);plt.ylim(0, 1)
for i, value in enumerate(yts[:, 0]):
    if value == 1:
        plt.axvline(x=i, ymin=0, ymax=0.5, linestyle='-', linewidth=2)

plt.subplot(5, 1, 3);plt.ylim(0, 1)
for i, value in enumerate(yts[:, 1]):
    if value == 1:
        plt.axvline(x=i, ymin=0, ymax=0.5, linestyle='-', linewidth=2)

plt.subplot(5, 1, 4);plt.ylim(0, 1)
for i, value in enumerate(yts[:, 2]):
    if value == 1:
        plt.axvline(x=i, ymin=0, ymax=0.5, linestyle='-', linewidth=2)

plt.subplot(5, 1, 5);plt.ylim(0, 1)
for i, value in enumerate(yts[:, 3]):
    if value == 1:
        plt.axvline(x=i, ymin=0, ymax=0.5, linestyle='-', linewidth=2) #  color='b',
plt.xlabel('Time')

plt.xlim(-1, 101)


plt.savefig('OU.svg')
plt.show()