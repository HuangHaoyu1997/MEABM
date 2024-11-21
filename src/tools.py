'''
Tools designed for LLM agents
'''

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

def adf_test(series: list[float]):
    '''
    检验序列series是否平稳
    input: list[float]
    output: bool, True 序列平稳, False 序列非平稳
    '''
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(pd.Series(series))
    # print('ADF Statistic:', result[0])
    # print('p-value:', result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print(f"p-value: {result[1]}, 拒绝原假设, 序列是平稳的")
        return True
    else:
        print(f"p-value: {result[1]}, 无法拒绝原假设, 序列是非平稳的")
        return False

def granger_causality_test(series1, series2, max_lag=12):
    '''
    检查series1与series2之间是否存在Granger因果关系
    '''
    from statsmodels.tsa.stattools import grangercausalitytests
    test_result = grangercausalitytests(series1, series2, maxlag=max_lag, verbose=False)

    # 输出结果
    for lag, results in test_result.items():
        p_values = [results[0][key][1] for key in results[0]]
        accept_flag = all(x < 0.05 for x in p_values)
        print(lag, p_values, accept_flag)