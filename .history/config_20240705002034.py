from time import time
class Configuration:
    seed = int(time()) # 123
    num_agents = 100
    num_time_steps = 12 * 50
    
    A = 1.0
    alpha_w = 0.05
    alpha_p = 0.10
    
    rn = 0.01
    pi_t = 0.02
    un = 0.04
    alpha_pi = 0.5
    alpha_u = 0.5
    
    gamma = 0.5
    beta = 0.5