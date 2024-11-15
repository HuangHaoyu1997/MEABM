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


def stdp_with_window(sa, sb, w, dt_positive=2, dt_negative=2, learning_rate=0.01, decay_factor=0.1, time_window=5):
    """
    sa: 突触前神经元A的脉冲序列 (list of int)
    sb: 突触后神经元B的脉冲序列 (list of int)
    w: 当前权重 (float)
    dt_positive: 突触前脉冲在突触后脉冲之前的时间窗口 (时间步数)
    dt_negative: 突触前脉冲在突触后脉冲之后的时间窗口 (时间步数)
    learning_rate: 学习率 (float)
    time_window: 有效时间窗口 (时间步数)
    
    return: 更新后的权重 (float)
    """

    def learn(sa, sb, w):
        # 获取脉冲时间戳
        spikes_A = [i for i in range(len(sa)) if sa[i] == 1]
        spikes_B = [i for i in range(len(sb)) if sb[i] == 1]

        if len(spikes_B) == 0:
            w -= decay_factor

        i, j = 0, 0  # 分别指向A和B的脉冲序列
        while i < len(spikes_A) and j < len(spikes_B):
            t_A, t_B = spikes_A[i], spikes_B[j]
            time_difference = t_B - t_A # 计算时间差
            
            # 正向STDP
            if 0 < time_difference <= dt_positive and time_difference <= time_window:
                w += learning_rate * (1.0 - time_difference / dt_positive)  # 使用线性衰减
                # w += learning_rate * np.exp(-time_difference / dt_positive)
                i += 1  # A的脉冲已经处理，移动到下一个脉冲
            elif time_difference > dt_positive:
                # A的脉冲在B的脉冲之前，并且超出了正向窗口，移动A的指针
                i += 1
            
            # 负向STDP
            else:  # time_difference <= 0
                if -dt_negative <= time_difference < 0 and -time_difference <= time_window:
                    w -= learning_rate * (1.0 + time_difference / dt_negative)  # 使用线性增加
                    # w -= learning_rate * np.exp(time_difference / dt_negative)
                j += 1 # 如果B的脉冲早于A的脉冲，移动B的指针
        w = max(0.0, min(w, 1.0))  # 假设权重在0到1之间
        return w
    # 在总发放脉冲序列上滑窗
    slide_window = 20 # 窗口大小
    steps = len(sa) // slide_window # 总步数
    for i in range(steps):
        start = i * slide_window
        end = (i + 1) * slide_window
        
        w = learn(sa[start:end], sb[start:end], w)
    return w

if __name__ == '__main__':
    
    # STDP_learner = STDP()
    
    # w = 0.0111  # 初始突触权重
    # T = 1000  # 总时间（毫秒）

    # # 记录尖峰事件
    # spikes_neuron1 = np.array(sorted(random.sample(range(1001), 20)))
    # spikes_neuron2 = spikes_neuron1 + 10 # sorted(random.sample(range(1001), 20))

    # # 进行STDP学习
    # for epoch in range(100):  # 多次迭代
    #     w = STDP_learner.learn(spikes_neuron1, spikes_neuron2, w)
    #     print(epoch, w)

    # print(f'最终突触权重: {w:.4f}')
    
    
    # 示例脉冲序列
    sa = [0, 1, 0, 0, 1, 0]  # 神经元A的脉冲序列
    sb = [0, 0, 1, 0, 0, 1]  # 神经元B的脉冲序列
    w = 0.5  # 初始权重

    # 更新权重
    updated_w = stdp_with_window(sa, sb, w, learning_rate=0.1)
    print(f"更新后的权重: {updated_w}")