# 事件驱动的脉冲编码概念神经元
class EventDrivenConceptNeuron:
    def __init__(self, v_th, m_init):
        self.v_th = v_th # firing threshold
        self.m = m_init # initial membrane potential
        
    def fire(self, input):
        if input - self.m > self.v_th:
            self.m = input
            return 1
        elif input - self.m < -self.v_th:
            self.m = input
            return -1
        else:
            return 0

class TransmissionNeuron:
    def __init__(self, v_th, m_init, m_tau, m_reset, beta):
        self.v_th = v_th # firing threshold
        self.m = m_init # initial membrane potential
        self.m_tau = m_tau # membrane time constant
        self.m_reset = m_reset # reset potential
        self.beta = beta # input scaling factor

    def fire(self, input):
        self.m = self.m * self.m_tau + input * self.beta

def encoding(input, concept_neurons: list[EventDrivenConceptNeuron]):
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