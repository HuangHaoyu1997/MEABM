import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from src.tools import *
np.random.seed(0)

# 加载数据
with open('./data/logs_intervention.pkl', 'rb') as f:
    data = pickle.load(f)
    # 'price', 'rate', 'production', 'imbalance', 'inflation_rate', 'taxes', 'unemployment_rate', 'deposit', 'avg_wage', 'GDP', 
    # 'capital', 'assets', 'gini'
    
    prices = [data[0][i]['price'] for i in range(len(data[0]))]
    intr_rate = [data[0][i]['rate'] for i in range(len(data[0]))]
    production = [data[0][i]['production'] for i in range(len(data[0]))]
    imbalance = [data[0][i]['imbalance'] for i in range(len(data[0]))]
    inflation_rate = [data[0][i]['inflation_rate'] for i in range(len(data[0]))]
    taxes = [data[0][i]['taxes'] for i in range(len(data[0]))]
    unemployment_rate = [data[0][i]['unemployment_rate'] for i in range(len(data[0]))]
    deposit = [data[0][i]['deposit'] for i in range(len(data[0]))]
    avg_wage = [data[0][i]['avg_wage'] for i in range(len(data[0]))]
    gdp = [data[0][i]['GDP']/1e6 for i in range(len(data[0]))]
    gini = [data[0][i]['gini'] for i in range(len(data[0]))]
    data_dict = {
        'price': prices,
        'rate': intr_rate,
        'production': production,
        'imbalance': imbalance,
        'inflation_rate': inflation_rate,
        'taxes': taxes,
        'unemployment_rate': unemployment_rate,
        # 'deposit': deposit,
        'avg_wage': avg_wage,
        'GDP': gdp,
        'gini': gini
    }
    for key in data_dict.keys():
        print(key, '\t', adf_test(data_dict[key]))
    

adf_test(pd.Series(production).diff().dropna())
data = pd.DataFrame(data_dict)
print(len(data['production']), len(data['GDP']))
# model = VAR(data)
# model = VECM(data, k_ar_diff=2, coint_rank=1)  # coint_rank=1表示一个协整关系
# vecm_fit = model.fit()
# var_fit = model.fit(maxlags=15)
# print(f"AIC: {var_fit.aic}")
# print(f"BIC: {var_fit.bic}")
# print(f"最佳滞后期 (AIC): {var_fit.k_ar}")
# 输出模型结果
# print(vecm_fit.summary())

# 查看短期动态
# short_term_effects = vecm_fit.params
# print("短期动态系数：")
# print(short_term_effects)

# 查看误差修正项
# error_correction_terms = vecm_fit.alpha
# print("误差修正项系数：")
# print(error_correction_terms)

from src.tools import granger_causality_test
for key in data_dict.keys():
    print(key)
    granger_causality_test(data[[key, 'production']], 3)
    print('\n\n\n')