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
    ConceptNeurons = [
        EventDrivenConceptNeuron(v_th=1, m_init=0),
        EventDrivenConceptNeuron(v_th=5, m_init=0),
        EventDrivenConceptNeuron(v_th=10, m_init=0),
        EventDrivenConceptNeuron(v_th=50, m_init=0),
    ]

    print(ConceptNeurons[0].m)
    output = encoding(25.3, ConceptNeurons)
    print(ConceptNeurons[0].m)