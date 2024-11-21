import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint

np.random.seed(0)

# 加载数据
with open('./data/logs_intervention.pkl', 'rb') as f:
    data = pickle.load(f)
    GDP = [data[0][i]['GDP']/1e6 for i in range(len(data[0]))]
    production = [data[0][i]['production']/5e2 for i in range(len(data[0]))]


def cointegration_test(series1, series2):
    '''
    no-cointegration test
    检查2个变量是否存在协整关系
    '''
    from statsmodels.tsa.stattools import coint
    score, p_value, _ = coint(series1, series2)
    if p_value < 0.05:
        print(p_value, "存在协整关系，考虑使用 VECM")
    else:
        print(p_value, "不存在协整关系，考虑使用 VAR")

data = pd.DataFrame({
    'GDP': GDP,
    'production': production
})

model = VECM(data, k_ar_diff=2, coint_rank=1)  # coint_rank=1表示一个协整关系
vecm_fit = model.fit()
# 输出模型结果
print(vecm_fit.summary())

# 查看短期动态
short_term_effects = vecm_fit.params
print("短期动态系数：")
print(short_term_effects)

# 查看误差修正项
error_correction_terms = vecm_fit.alpha
print("误差修正项系数：")
print(error_correction_terms)



# ADF检验函数
def adf_test(series):
    result = adfuller(pd.Series(series))
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print("拒绝原假设：序列是平稳的")
    else:
        print("无法拒绝原假设：序列是非平稳的")

# 执行ADF检验
adf_test(production)
adf_test(GDP)

data = data.dropna()
# 拟合VAR模型
model = VAR(data)
results = model.fit(maxlags=15)  # 设置一个较大的最大滞后期

print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
print(f"最佳滞后期 (AIC): {results.k_ar}")

data = pd.DataFrame({
    'GDP': GDP,
    'production': production
})

# 进行格兰杰因果关系检验
max_lag = 5  # 最大滞后期
test_result = grangercausalitytests(data[['production', 'GDP']], 5, verbose=False)

# 输出结果
for lag, results in test_result.items():
    print(f"Lag {lag}:")
    print(results[1][0].aic, results[1][0].bic, results[1][1].aic, results[1][1].bic,)

    for key, value in results[0].items():
        print(f"  {key} p-value: {value[1]}")
