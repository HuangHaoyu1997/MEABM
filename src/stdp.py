import numpy as np

class STDP:
    def __init__(self, A_plus=0.005, tau_pos=20.0, A_minus=0.005, tau_neg=20.0):
        # STDP参数
        self.tau_pos = tau_pos  # 突触增强的时间常数（毫秒）
        self.tau_neg = tau_neg  # 突触减弱的时间常数（毫秒）
        self.A_plus = A_plus  # 突触增强的幅度
        self.A_minus = A_minus  # 突触减弱的幅度

    def stdp_learning(self, spikes1, spikes2, w):
        '''
        Neuron-to-Neuron STDP Learning
        :param spikes1: spikes of neuron1
        :param spikes2: spikes of neuron2
        :param w: initial weight
        :return: updated weight
        '''
        for t1 in spikes1:
            for t2 in spikes2:
                if t1 < t2:  # neuron1 spike before neuron2
                    delta_t = t2 - t1
                    w += self.A_plus * np.exp(-delta_t / self.tau_pos)  # 突触增强
                elif t1 > t2:  # neuron1 spike after neuron2
                    delta_t = t1 - t2
                    w -= self.A_minus * np.exp(-delta_t / self.tau_neg)  # 突触减弱
        # weight clipping
        w = np.clip(w, 0, 1)
        return w