from src.srn import EventDrivenConceptNeuron, PECN, NECN, TransmissionNeuron, encoding
import numpy as np
import matplotlib.pyplot as plt

length = 200
simu_GDP = np.sin(np.arange(0, 20, 0.1)) * 200

# ConceptNeurons = [
#         EventDrivenConceptNeuron(v_th=1, m_init=0, polar=-1),
#         EventDrivenConceptNeuron(v_th=15, m_init=0, polar=-1),
#         EventDrivenConceptNeuron(v_th=10, m_init=0, polar=-1),
#         EventDrivenConceptNeuron(v_th=50, m_init=0, polar=-1),
#     ]
ConceptNeurons = [
        PECN(v_th=1, m_init=0),
        PECN(v_th=15, m_init=0),
        EventDrivenConceptNeuron(v_th=10, m_init=0, polar=-1),
        EventDrivenConceptNeuron(v_th=50, m_init=0, polar=-1),
    ]
yts = []
for xt in simu_GDP:
    yt, ConceptNeurons = encoding(xt, ConceptNeurons)
    yts.append(yt)
yts = np.array(yts) # (200, 4)
print(yts.shape)


tran_neurons = [
    TransmissionNeuron(v_th=1.0, m_init=0, m_tau=0.3, m_reset=0),
    TransmissionNeuron(v_th=1.0, m_init=0, m_tau=0.4, m_reset=0),
    TransmissionNeuron(v_th=1.0, m_init=0, m_tau=0.5, m_reset=0),
    TransmissionNeuron(v_th=1.0, m_init=0, m_tau=0.6, m_reset=0),
    ]

W = np.random.rand()

tran_neuron = TransmissionNeuron(v_th=1.0, m_init=0, m_tau=0.5, m_reset=0)

m_potentials = []
Ws = np.zeros((length)) # weight history
post_spikes = np.zeros((length)) # spike history

for t in range(length):
    spike = tran_neuron.fire(yts[t,0] * W)
    post_spikes[t] = spike
    W += 0.04 * max(yts[t,0], 0)
    W -= 0.04 * spike
    W = max(min(W, 1), -1)
    Ws[t] = W
    m_potentials.append(tran_neuron.m)
    print(W, spike, tran_neuron.m)
print([i.m_tau for i in tran_neurons])
plt.figure(1)
plt.subplot(611); plt.plot(simu_GDP, '.')
plt.subplot(612); plt.plot(yts[:, 0], '.')
plt.subplot(613); plt.plot(yts[:, 1], '.')
plt.subplot(614); plt.plot(Ws)
plt.subplot(615); plt.plot(post_spikes, '.')
plt.subplot(616); plt.plot(m_potentials)
plt.show()