import numpy as np
import random

def stdp(weight: float, A_pre: float, A_post: float, decay: float, pre_spike: bool, post_spike: bool) -> float:
    w = weight
    if pre_spike:
        w += A_pre
    if post_spike:
        w -= A_post
    if not pre_spike and not post_spike:
        w -= decay
    w = max(min(w, 1), -1)
    return w

class STDP:
    def __init__(self, A_plus=0.005, tau_pos=20.0, A_minus=0.005, tau_neg=20.0):
        # STDP参数
        self.tau_pos = tau_pos  # 突触增强的时间常数（毫秒）
        self.tau_neg = tau_neg  # 突触减弱的时间常数（毫秒）
        self.A_plus = A_plus  # 突触增强的幅度
        self.A_minus = A_minus  # 突触减弱的幅度

    def learn(self, spikes1, spikes2, w):
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

if __name__ == '__main__':
    
    STDP_learner = STDP()
    
    w = 0.0111  # 初始突触权重
    T = 1000  # 总时间（毫秒）

    # 记录尖峰事件
    spikes_neuron1 = np.array(sorted(random.sample(range(1001), 20)))
    spikes_neuron2 = spikes_neuron1 + 10 # sorted(random.sample(range(1001), 20))

    # 进行STDP学习
    for epoch in range(100):  # 多次迭代
        w = STDP_learner.learn(spikes_neuron1, spikes_neuron2, w)
        print(epoch, w)

    print(f'最终突触权重: {w:.4f}')