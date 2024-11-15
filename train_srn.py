from src.srn import EventDrivenConceptNeuron, PECN, NECN, TransmissionNeuron, encoding
from src.stdp import stdp, stdp_with_window
import numpy as np
import matplotlib.pyplot as plt
from src.utils import moving_average

import pickle

with open('./data/logs_intervention.pkl', 'rb') as f:
    logs = pickle.load(f)
GDP = [logs[0][i]['GDP']/1e6 for i in range(len(logs[0]))]
GDP_MA = moving_average(GDP, 20)

simu_GDP = GDP_MA

length = len(simu_GDP)
# simu_GDP = np.sin(np.arange(0, 10, 0.1)) * 15
###################################################################



ConceptNeurons = [
        NECN(v_th=0.2),
        NECN(v_th=5),
        NECN(v_th=20),
    ]

yts = []
for xt in simu_GDP:
    yt, ConceptNeurons = encoding(xt, ConceptNeurons)
    yts.append(yt)
yts = np.array(yts) # (200, 4)
print('yts.shape: ', yts.shape)


tran_neurons = [
    TransmissionNeuron(v_th=0.8, m_init=0, m_tau=0.1, m_reset=0),
    TransmissionNeuron(v_th=0.8, m_init=0, m_tau=0.3, m_reset=0),
    TransmissionNeuron(v_th=0.8, m_init=0, m_tau=0.5, m_reset=0),
    TransmissionNeuron(v_th=0.8, m_init=0, m_tau=0.7, m_reset=0),
    TransmissionNeuron(v_th=0.8, m_init=0, m_tau=0.9, m_reset=0),
    ]

Ws = []

plt.figure(1, figsize=(8, 12))
plt.subplot(7, 1, 1); plt.plot(yts[:, 0], '.');   plt.title('Concept Neuron vth=1')
plt.subplot(7, 1, 2); plt.plot(GDP_MA, '*');    plt.title('GDP')
for i, neuron in enumerate(tran_neurons):
    W = 1.0
    spikes = [neuron.fire(yts[t,0] * W) for t in range(length)]
    W_ = stdp_with_window(sa=yts[:, 0], sb=spikes, w=W, learning_rate=0.01, decay_factor=0.01)
    Ws.append(W_)
    print(W, W_)
    plt.subplot(7, 1, i+3); plt.plot(spikes, '.'); plt.title('Spiking history')
plt.tight_layout()
plt.show()


plt.figure(2, figsize=(8, 12))
for i, neuron in enumerate(tran_neurons):
    spikes = [neuron.fire(yts[t,0] * Ws[i]) for t in range(length)]
    plt.subplot(5, 1, i+1); plt.plot(spikes, '.'); plt.title('Spiking history')
plt.tight_layout()
plt.show()


# for i in range(10):

#     W = np.random.rand()*0.5+0.5
#     m_potentials = [] # membrane potential history
#     Ws = []           # weight history
#     post_spikes1 = []  # spike history
#     for t in range(length):
#         spike = tran_neuron.fire(yts[t,0] * W)
#         post_spikes1.append(spike)
#         # W = stdp(W, 0.04, 0.04, 0.002, yts[t, 0], spike)
        
#         # Ws.append(W)
#         m_potentials.append(tran_neuron.m)
#     W_ = stdp_with_window(sa=yts[:, 0], sb=post_spikes1, w=W, learning_rate=0.01)

#     post_spikes2 = []
#     for t in range(length):
#         spike = tran_neuron.fire(yts[t,0] * W_)
#         post_spikes2.append(spike)
#     W__ = stdp_with_window(sa=yts[:, 0], sb=post_spikes2, w=W_, learning_rate=0.01)

#     print(W, W_, W__)

# plt.figure(1, figsize=(8, 12))
# plt.subplot(411); plt.plot(yts[:, 0], '.');   plt.title('Concept Neuron vth=1')
# plt.subplot(412); plt.plot(post_spikes1, '.'); plt.title('Spiking history')
# plt.subplot(413); plt.plot(m_potentials);     plt.title('Membrane potential of transmission neuron')
# plt.subplot(414); plt.plot(post_spikes2, '.'); plt.title('Spiking history')
# plt.tight_layout()
# plt.show()

# print([i.m_tau for i in tran_neurons])


# plt.figure(1, figsize=(8, 12))
# plt.subplot(611); plt.plot(simu_GDP, '.');    plt.title('GDP')
# plt.subplot(612); plt.plot(yts[:, 0], '.');   plt.title('Concept Neuron vth=1')
# plt.subplot(613); plt.plot(yts[:, 1], '.');   plt.title('Concept Neuron vth=5')
# plt.subplot(614); plt.plot(Ws);               plt.title('Weight update history')
# plt.subplot(615); plt.plot(post_spikes, '.'); plt.title('Spiking history')
# plt.subplot(616); plt.plot(m_potentials);     plt.title('Membrane potential of transmission neuron')

# plt.tight_layout()
# plt.show()