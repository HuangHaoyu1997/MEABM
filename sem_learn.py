import pandas as pd
from semopy import Model

# 创建示例数据
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Y': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}

df = pd.DataFrame(data)

# 定义初步模型
model_desc = """
Y ~ X1 + X2
"""

# 创建模型
model = Model(model_desc)

# 拟合数据
model.fit(df)

# 输出结果
results = model.inspect()
print(results)