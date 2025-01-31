'''
opinion dynamics of agents in a macroeconomic agent-based model (MEABM)
'''
from agent import agent
import numpy as np
import random

def Deffuant_Weisbuch(A: agent, B: agent, opinion: str, eps: float, mu: float):
    '''
    Deffuant-Weisbuch model of opinion dynamics

    A: random selected agent
    B: random selected agent
    opinion: the opinion waiting to be updated, work_propensity, consume_propensity
    eps: bounded confidence parameter
    mu: influence parameter
    '''
    if opinion == 'w': # work_propensity
        if abs(A.pw - B.pw) < eps:
            if random.random() < 0.95: # 大概率观点融合
                A_opinion = (1 - mu) * A.pw + mu * (B.pw - A.pw)
                B_opinion = (1 - mu) * B.pw + mu * (A.pw - B.pw)
            else: # 小概率观点极化
                A_opinion = (1 - mu) * A.pw + mu * (A.pw - B.pw)
                B_opinion = (1 - mu) * B.pw + mu * (B.pw - A.pw)
            A.pw = A_opinion
            B.pw = B_opinion
            return True
    elif opinion == 'c': # consume_propensity
        if abs(A.pc - B.pc) < eps:
            if random.random() < 0.95: # 大概率观点融合
                A_opinion = (1 - mu) * A.pc + mu * (B.pc - A.pc)
                B_opinion = (1 - mu) * B.pc + mu * (A.pc - B.pc)
            else: # 小概率观点极化
                A_opinion = (1 - mu) * A.pc + mu * (A.pc - B.pc)
                B_opinion = (1 - mu) * B.pc + mu * (B.pc - A.pc)
            A.pc = A_opinion
            B.pc = B_opinion
            return True
    else:
        return False
    
    # return A_opinion, B_opinion


if __name__ == '__main__':
    A = agent(1, 0.65, 0.23, 0, 0, 0, 0)
    B = agent(2, 0.72, 0.33, 0, 0, 0, 0)
    print(A.pw, B.pw)
    print(Deffuant_Weisbuch(A, B, 'w', 0.1, 0.2))
    print(A.pw, B.pw)
