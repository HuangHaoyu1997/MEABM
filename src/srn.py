import numpy as np

# 事件驱动的脉冲编码概念神经元
class PECN:
    '''
    Positive Event-driven Concept Neuron (P-ECN)
    '''
    def __init__(self, v_th, m_init=0):
        self.v_th = v_th
        self.m = m_init
    def fire(self, input):
        if input - self.m > self.v_th:
            self.m = input
            spike = 1
        else:
            spike = 0
        if input < self.m:
            self.m = input
        return spike

class NECN:
    '''
    Negative Event-driven Concept Neuron (N-ECN)
    '''
    def __init__(self, v_th, m_init=0):
        self.v_th = v_th
        self.m = m_init
    def fire(self, input):
        if input - self.m < -self.v_th:
            self.m = input
            spike = 1
        else:
            spike = 0
        if input > self.m:
            self.m = input
        return spike

class EventDrivenConceptNeuron:
    def __init__(self, v_th, m_init, polar=1):
        '''
        polar: 1 for increasing events, -1 for decreasing events
        '''
        self.polar = polar
        self.v_th = v_th # firing threshold
        self.m = m_init # initial membrane potential
        
    def fire(self, input):
        if self.polar == 1:
            # 输入信号-膜电位大于阈值，脉冲发生
            if input - self.m > self.v_th: spike = 1
            else:                          spike = 0
        elif self.polar == -1:
            # 输入信号小于膜电位，且输入信号与膜电位差值大于阈值，脉冲发生
            if input < self.m and abs(input - self.m) > self.v_th: spike = 1
            else:                                                  spike = 0
        self.m = input # 膜电位更新
        return spike

class TransmissionNeuron:
    def __init__(self, v_th, m_init, m_tau, m_reset=0, beta=0.5005):
        '''
        beta: input scaling factor
        '''
        self.v_th = v_th # firing threshold
        self.m = m_init # initial membrane potential
        self.m_tau = m_tau # membrane time constant
        self.m_reset = m_reset # reset potential
        self.beta = beta # input scaling factor

    def fire(self, input):
        self.m = self.m * self.m_tau + input * self.beta
        if self.m - self.m_reset > self.v_th:
            self.m = self.m_reset
            return 1
        else:
            return 0

def encoding(input, concept_neurons: list[EventDrivenConceptNeuron|PECN|NECN]):
    encoding = [neuron.fire(input) for neuron in concept_neurons]
    return encoding, concept_neurons



if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt

    ConceptNeurons = [
        EventDrivenConceptNeuron(v_th=1, m_init=0),
        EventDrivenConceptNeuron(v_th=5, m_init=0),
        EventDrivenConceptNeuron(v_th=10, m_init=0),
        EventDrivenConceptNeuron(v_th=50, m_init=0),
    ]

    print(ConceptNeurons[0].m)
    output = encoding(25.3, ConceptNeurons)
    print(ConceptNeurons[0].m)

    name = '20241222'
    with open(f'/home/hhy/MEABM/data/logs_PSRN_{name}.pkl', 'rb') as f:
        logs_PSRN = pickle.load(f)
    
    # series = np.array([[log[key]['GDP'] for key in log.keys()] for log in logs_PSRN]) / 1e5
    series = np.array([[log[key]['unemployment_rate'] for key in log.keys()] for log in logs_PSRN])
    print(series.shape)
    plt.subplot(5, 1, 1)
    plt.plot(series[0,:])  # np.linspace(0, n_steps * dt, n_steps)
    plt.grid(True)
    plt.ylabel('GDP')

    ConceptNeurons = [
            # NECN(v_th=0.01),
            # NECN(v_th=0.05),
            # NECN(v_th=0.10),
            # NECN(v_th=0.2),
            PECN(v_th=0.009),
            PECN(v_th=0.028),
            PECN(v_th=0.05),
            PECN(v_th=0.10),
        ]
    yts = []
    for xt in series[0,:]:
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


    plt.savefig('/home/hhy/MEABM/data/GDP-Concept-Neuron.png')
    plt.show()