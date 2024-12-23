import pickle
import random
from src.utils import ornstein_uhlenbeck_process, plot_one_fig, total_deposit
import matplotlib.pyplot as plt
import numpy as np
from config import Configuration

if __name__ == '__main__':
    np.random.seed(123)
    random.seed(123)

    name = '20241222'
    with open(f'data/logs_PSRN_{name}.pkl', 'rb') as f:
        logs_PSRN = pickle.load(f)
    with open(f'data/logs_Taylor_{name}.pkl', 'rb') as f:
        logs_Taylor = pickle.load(f)
    with open(f'data/logs_Fixed_{name}.pkl', 'rb') as f:
        logs_Fixed = pickle.load(f)

    plot_one_fig(
        np.array([[log[key]['price'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['price'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['price'] for key in log.keys()] for log in logs_Fixed]), 
        name='Price of Goods')
    
    plot_one_fig(
        np.array([[log[key]['production'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['production'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['production'] for key in log.keys()] for log in logs_Fixed]), 
        name='Production of Goods')
    
    plot_one_fig(
        np.array([[log[key]['gini'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['gini'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['gini'] for key in log.keys()] for log in logs_Fixed]), 
        name='Gini Coefficient')
    
    plot_one_fig(
        np.array([[log[key]['inflation_rate'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['inflation_rate'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['inflation_rate'] for key in log.keys()] for log in logs_Fixed]), 
        name='Inflation Rate')

    plot_one_fig(
        np.array([[log[key]['unemployment_rate'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['unemployment_rate'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['unemployment_rate'] for key in log.keys()] for log in logs_Fixed]), 
        name='Unemployment Rate')
    
    plot_one_fig(
        np.array([[log[key]['GDP'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['GDP'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['GDP'] for key in log.keys()] for log in logs_Fixed]), 
        name='Nomial GDP')
    
    plot_one_fig(
        np.array([[log[key]['imbalance'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['imbalance'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['imbalance'] for key in log.keys()] for log in logs_Fixed]), 
        name='Supply-demand Imbalance')

    plot_one_fig(
        np.array([[log[key]['avg_wage'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['avg_wage'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['avg_wage'] for key in log.keys()] for log in logs_Fixed]), 
        name='Average Wage per Capita')
    
    plot_one_fig(
        np.array([[np.std(log[key]['wage']) for key in log.keys()] for log in logs_PSRN]), 
        np.array([[np.std(log[key]['wage']) for key in log.keys()] for log in logs_Taylor]), 
        np.array([[np.std(log[key]['wage']) for key in log.keys()] for log in logs_Fixed]), 
        name='Standard Deviation of Wage')
    
    plot_one_fig(
        np.array([[log[key]['rate'] for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['rate'] for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['rate'] for key in log.keys()] for log in logs_Fixed]), 
        name='Interest Rate')
    
    plot_one_fig(
        np.array([[total_deposit(log[key]['deposit'])/Configuration().num_agents for key in log.keys()] for log in logs_PSRN]), 
        np.array([[total_deposit(log[key]['deposit'])/Configuration().num_agents for key in log.keys()] for log in logs_Taylor]), 
        np.array([[total_deposit(log[key]['deposit'])/Configuration().num_agents for key in log.keys()] for log in logs_Fixed]), 
        name='Average Deposit per Capita')
    
    plot_one_fig(
        np.array([[log[key]['taxes']/Configuration().num_agents for key in log.keys()] for log in logs_PSRN]), 
        np.array([[log[key]['taxes']/Configuration().num_agents for key in log.keys()] for log in logs_Taylor]), 
        np.array([[log[key]['taxes']/Configuration().num_agents for key in log.keys()] for log in logs_Fixed]), 
        name='Avg tax revenue per capita')

    # theta = 0.9  # Speed of mean reversion
    # mu = 4.63     # Long-term mean
    # sigma = 2.5  # Volatility
    # x0 = 0.5     # Initial value
    # ou_process = ornstein_uhlenbeck_process(theta, mu, sigma, x0, dt=0.1, n_steps=401)