agent_granger_causality = '''
# 任务描述
我有以下经济变量：{variables}。请使用`granger_causality`函数检验这些变量之间的两两因果关系，并基于检验结果构建一个因果图。

# 步骤
1. **变量列表**:
    - {variables}

2. **因果关系检验**:
    - 对于每对变量 (A, B)，执行以下操作：
        - 检验 A 是否 Granger 导致 B。
        - 检验 B 是否 Granger 导致 A。
    - 使用`granger_causality(A, B)`和`granger_causality(B, A)`来进行检验。

3. **汇总结果**:
    - 列出所有显著的因果关系。例如：
        - A → B
        - C → D

4. **构建因果图**:
    - 根据上述因果关系，绘制因果图。
    - 可以使用图形化工具（如NetworkX、Graphviz）生成图像，或以邻接矩阵形式表示。

# 示例

**输入变量**:
- A, B, C, D

**因果关系检验结果**:
- A Granger 导致 B
- C Granger 导致 D

**因果图**:
- A → B
- C → D

# 注意事项
- 确保每个`granger_causality`调用的滞后阶数（lag）适当选择。
- 检验结果的显著性水平（如p值）需满足预设标准（如p < 0.05）才认为存在因果关系。
- 避免自环和多重边，以保持因果图的清晰性。

# 工具调用示例
```python
import networkx as nx
import matplotlib.pyplot as plt

# 假设已获得因果关系列表 causal_relations = [('A', 'B'), ('C', 'D')]

G = nx.DiGraph()
G.add_edges_from(causal_relations)

plt.figure(figsize=(8,6))
nx.draw(G, with_labels=True, node_color='lightblue', arrows=True)
plt.title('因果图')
plt.show()
'''

'''
### 使用指南

1. **准备变量列表**: 将所有需要检验的经济变量列出，并替换模板中的`{variables}`。

2. **执行因果关系检验**: 按照模板中的步骤，使用`granger_causality`函数对每对变量进行双向检验，并记录显著的因果关系。

3. **汇总并构建因果图**: 将所有显著的因果关系整理后，使用代码示例中的方法或其他图形化工具生成因果图。

4. **调整与优化**:
    - 根据具体数据情况，调整`granger_causality`函数的参数，如滞后阶数。
    - 如果变量数量较多，可以考虑自动化上述步骤，编写脚本批量处理。

### 扩展建议

- **自动化脚本**: 可以编写一个Python脚本，自动遍历所有变量对，调用`granger_causality`函数，并根据结果生成因果关系列表和因果图。

- **结果解释**: 在因果图生成后，可以进一步对图中的因果关系进行经济学解释，探讨变量之间的潜在机制。

- **动态更新**: 随着新的数据或变量的加入，及时更新因果图，保持分析的时效性和准确性。

通过以上提示框架，您可以有效地控制LLM执行因果关系检验，并构建经济变量之间的因果图。如果需要更详细的代码实现或进一步的指导，请随时告知！
'''