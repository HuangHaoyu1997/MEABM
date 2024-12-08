'''
Tools designed for LLM agents
'''
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from api_key import zhipu_api_key
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
import pickle

@tool
def add(a: float, b: float) -> float:
    '''Add two numbers.'''
    return a + b

@tool
def sub(a: float, b: float) -> float:
    '''Subtract two numbers.'''
    return a - b

@tool
def mul(a: float, b: float) -> float:
    '''Multiply two numbers.'''
    return a * b

@tool
def div(a: float, b: float) -> float:
    '''Divide two numbers.'''
    return a / b
@tool
def pow(a: float, b: float) -> float:
    '''Power two numbers.
    a is the base number, b is the exponent.'''
    return a ** b

@tool
def today() -> str:
    '''Get today's date in string format.'''
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d')

@tool
def wiki(query: str) -> str:
    '''Search for a query on Wikipedia and return the top 1 result.'''
    from langchain.utilities import WikipediaAPIWrapper
    wikipedia = WikipediaAPIWrapper(top_k_results=1)
    return wikipedia.run(query)

@tool
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

@tool
def adf_test_name(variable_name: str):
    '''
    检验变量`variable_name`是否平稳
    input: str, 变量名称
    output: bool, True表示变量平稳, False表示变量非平稳
    '''
    series = data_dict[variable_name]
    return adf_test(series)
    
def adf_test(series: list[float]):
    '''
    检验序列series是否平稳
    input: list[float]
    output: bool, True表示序列平稳, False表示序列非平稳
    '''
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(pd.Series(series))
    # print('ADF Statistic:', result[0])
    # print('p-value:', result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print(f'   {key}: {value}')
    
    if result[1] <= 0.05: return True # f"p-value: {result[1]}, 拒绝原假设, 序列是平稳的"
    else:                 return False # f"p-value: {result[1]}, 无法拒绝原假设, 序列是非平稳的"

@tool
def granger_causality_test(series, max_lag=12):
    '''
    检查series1与series2之间是否存在Granger因果关系
    '''
    from statsmodels.tsa.stattools import grangercausalitytests
    # print(max_lag)
    test_result = grangercausalitytests(series, maxlag=max_lag, verbose=False)

    # 输出结果
    for lag, results in test_result.items():
        p_values = [results[0][key][1] for key in results[0]]
        accept_flag = all(x < 0.05 for x in p_values)
        print(lag, p_values, accept_flag)

if __name__ == '__main__':

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

    chatbot = ChatZhipuAI(api_key=zhipu_api_key, model='glm-4-plus', streaming=False)
    zhipu_with_tools = chatbot.bind_tools([add, sub, mul, div, pow, today, wiki, adf_test_name])
    tools_list = [item['function']['name'] for item in zhipu_with_tools.kwargs['tools']]

    query = '变量GDP是时间平稳序列吗？'
    results = zhipu_with_tools.invoke(query)
    if results.tool_calls:
        for tool_call in results.tool_calls:
            selected_tool = {
                'add': add, 
                'sub': sub, 
                'mul': mul, 
                'div': div, 
                'pow': pow, 
                'today': today, 
                'wiki': wiki,
                'adf_test_name': adf_test_name,
                }[tool_call['name']]
            output = selected_tool.invoke(tool_call['args'])
            # print(output)
            messages = [
                SystemMessage(content='你是一个精通经济调控的经济学专家。'), # ，能够有效分析各类经济变量的变化，给出科学的经济调控建议。
                HumanMessage(content=f'{query}. 以下是从函数 \'{tool_call["name"]}\'得到的结果: {output}. \n 请给出你的回答。'),
            ]
            print(chatbot.invoke(messages).content)
            
    else:
        print(results.content)

