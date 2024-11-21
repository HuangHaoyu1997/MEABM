import pandas as pd
import numpy as np

from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore, ExhaustiveSearch
from pgmpy.inference import VariableElimination

# 确保时间序列数据是格式化的，通常以 Pandas DataFrame 的形式存储。假设你有两个时间序列：X 和 Y。
data = {
    'X': np.sin(np.linspace(0, 2*np.pi, 100)),
    'Y': np.sin(np.linspace(0, 2*np.pi, 100)) + 2,
}
df = pd.DataFrame(data)

# 离散化数据
# 贝叶斯网络通常处理离散变量。如果你的数据是连续的，可以考虑将其离散化。
df['X'] = pd.cut(df['X'], bins=20, labels=[str(i+1) for i in range(20)]) # 
df['Y'] = pd.cut(df['Y'], bins=20, labels=[str(i+1) for i in range(20)]) # 
# print(df['X_disc'])


# 定义贝叶斯网络结构
# 定义变量之间的因果关系。可以手动指定，也可以使用一些算法（如贪心算法）来学习结构。
model = BayesianNetwork([('X', 'Y')])  # 假设 X 影响 Y
# model.fit(df[['X', 'Y']], estimator=MaximumLikelihoodEstimator)
# inference = VariableElimination(model)
# result = inference.query(variables=['Y'], evidence={'X': '16'})
# print(result)


# 连续变量
np.random.seed(42)
n_samples = 1000
X = np.random.normal(0, 1, n_samples)
Y = 2 * X + np.random.normal(0, 1, n_samples)  # Y 与 X 存在线性关系
data = pd.DataFrame(data={'X': X, 'Y': Y})



# 使用结构学习算法（例如：爬山算法）
# estimator = HillClimbSearch(data)
estimator = ExhaustiveSearch(data)
model = estimator.estimate() # scoring_method=BicScore(data)
print("Learned structure:", model.edges())

# 定义高斯贝叶斯网络
model = LinearGaussianBayesianNetwork(model.edges())
model.fit(data)

inference = VariableElimination(model)

y_cpd = model.get_cpds('Y')

# 提取均值和标准差
mean_y = y_cpd
print(mean_y)
# 进行因果推断，给定 X 的值
query_result = inference.query(variables=['Y'], evidence={'X': 1.0})

print(query_result)