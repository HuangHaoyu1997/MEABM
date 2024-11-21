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
    GDP = [data[0][i]['GDP']/1e6 for i in range(len(data[0]))]
    production = [data[0][i]['production']/5e2 for i in range(len(data[0]))]

adf_test(pd.Series(production).diff().dropna())
data = pd.DataFrame({
    # 'GDP': pd.Series(GDP).dropna(),
    # 'production': pd.Series(production).diff().dropna()
    'GDP': GDP,
    'production': production
})
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

