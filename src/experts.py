from config import Configuration
from src.api_key import zhipu_api_key
from src.utils import img2base64
import cv2, base64, requests
import numpy as np
import pickle
from zhipuai import ZhipuAI



def get_expert_response(img, sys_prompt:str, user_prompt:str, config:Configuration):
    
    _, encoded_image = cv2.imencode('.png', img)
    binary_image = encoded_image.tobytes()
    base64_image = base64.b64encode(binary_image).decode('utf-8')
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": config.gpt_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": 
                [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ]
                }
            ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import split_img
    
    # config = Configuration()
    
    # subfigs = split_img('./figs/bar-event-intervention.png')
    # fig_names = [
    #     'price', 
    #     'interest rate ', 
    #     'employment rate', 
    #     'inflation rate', 
    #     'gini coefficient',
    #     'demand-supply imbalance', 
    #     'firm\'s capital and bank\'s assets', 
    #     'goods production', 
    #     'GDP', 
    #     'deposit per capita',
    #     'wage per capita',
    #     'tax revenue per capita',
    #     'standard deviation of workers\' wages',
    #     ]
    # for fig_name, subfig in zip(fig_names, subfigs):
    #     prefix = f'This is the simulated curve of {fig_name} from the agent-based economic model.'
    #     response = get_expert_response(subfig, config)
    #     print(response)
    
    
    
    # You are an expert economist with a strong background in economic theory and empirical analysis. 
    # 你面对的是一个具有{config.num_agents}个消费者/工人，1个消费品生产企业，1个储蓄银行，1个政府构成的小型模拟经济体。
    # 消费者有Individual的就业意愿和消费意愿，并根据市场变化加以调整。
    # 企业招聘工人生产消费品，支付工资，出售消费品换取利润，并根据供需信号调整生产计划。
    # 银行有储蓄账户，每年向消费者支付利息一次，并根据市场变化调整存款利息。
    # 政府对工资收入和企业经营收入征税，并立即均分给消费者。

    

    
    # 你需要分析模拟经济体的一项（以图表展示的）经济指标，并给出分析意见。
    
    # 你需要综合所有专家对每一个图表（经济指标）的分析结果，给出对模拟经济体的总体分析意见。
    
    
    
    # 告知event
    
    # 每个专家对各自图表进行分析
    
    # 专家汇总分析意见
    
    # 给出干预选项和调控目标，每个专家根据总的分析意见给出干预决策
    
    # 汇总干预决策，形成最终决策方案
    
    # 执行决策，反馈运行结果
    
    # from prompts import user_prompt
    # user_prompt.format(fig_name='price of goods',
    #                    now_step='400',
    #                    index_name='price',
    #                    event_name='economic crisis',
    #                    event_time='350',
    #                    event_effect='workers\' work will be reduced',)
    

    client = ZhipuAI(api_key=zhipu_api_key)

    name = '20241222'
    with open(f'data/logs_PSRN_{name}.pkl', 'rb') as f:
        logs_PSRN = pickle.load(f)
    data = np.array([[log[key]['production'] for key in log.keys()] for log in logs_PSRN])
    data_str = np.array2string(data[0], separator=', ')
    response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请你分析这个时序变量的变化趋势。"},
                {"type": "image_url", "image_url": {"url": img2base64("/home/hhy/MEABM/产量.png")}}
            ],
    # [{"type": "text", "text": '请你分析这个时序变量的变化趋势。'+data_str}],
        },]
    )
    print(response.choices[0].message.content)

    从图中可以看出，该时序变量在600个月内的变化趋势如下：
    1. **初期波动**：在开始阶段，变量显示出较大的波动，数据点在20000至25000之间波动。
    2. **中期趋势**：在200个月左右，变量达到一个高峰，接近25000。随后，变量开始逐渐下降，波动幅度减小。
    3. **后期稳定**：在400个月之后，变量趋于稳定，波动范围缩小，主要在20000上下浮动。
    总体来看，该变量在初期波动较大，中期经历了一个高峰后逐渐下降，后期趋于稳定。

    要分析这个时序变量的变化趋势，我们可以从以下几个方面进行：
    1. **总体趋势**
    首先，观察数据的变化范围和总体趋势。数据从10000开始，呈现出波动上升的趋势，中间有几次明显的波动，但整体上是增加的。
    2. **波动性**
    数据的波动性较大，存在多次上下波动。可以通过计算标准差来量化这种波动性。
    3. **阶段性分析**
    可以将数据分为几个阶段来分析：
    - **初期阶段**（前20-30个数据点）：数据从10000开始，逐渐上升，波动较小。
    - **中期阶段**（中间部分）：数据波动较大，有几次明显的上升和下降。
    - **后期阶段**（后20-30个数据点）：数据整体上仍然呈现上升趋势，但波动性依然存在。
    # 4. **极值分析**
    # 找出数据中的最大值和最小值，以及它们出现的位置，可以帮助我们理解数据的极端情况。
    # - 最大值：约26096.7
    # - 最小值：约16205.9
    # 5. **均值和趋势线**
    # 计算数据的均值，并绘制趋势线（如移动平均线），可以帮助我们更清晰地看到数据的长期趋势。
    # 6. **周期性分析**
    # 观察数据是否有周期性变化。虽然从直观上看不太明显，但可以通过傅里叶变换等工具进行更详细的分析。
    # 7. **增长率分析**
    # 计算不同时间段的数据增长率，可以帮助我们理解数据的增长速度。
    # 具体步骤：
    # 1. **绘制时间序列图**：将数据点绘制在时间轴上，直观地观察变化趋势。
    # 2. **计算统计指标**：均值、标准差、最大值、最小值等。
    # 3. **移动平均线**：计算并绘制移动平均线，平滑数据波动，观察长期趋势。
    # 4. **周期性分析**：使用傅里叶变换或其他工具检测周期性成分。
    # 5. **增长率计算**：计算不同时间段的增长率。

    # 通过以上分析和可视化，可以更全面地理解这个时序变量的变化趋势。具体结论需要根据实际数据和业务背景进行进一步解读。
    
    # system_prompt= '''
    # 你是一个专业的经济学家，你正在面对的是一个多智能体建模的宏观经济仿真模型。注意，这个模型不一定和真实经济系统遵守一样的经济规律。
    # 这个仿真模型由个体智能体、一个消费品生产企业、一个储蓄银行和一个政府构成。
    # 个体智能体：依照个体的就业意愿，以概率在消费品生产企业工作，获得工资，并支付个人所得税。依据个体的消费意愿，花费自己存款的部分购买消费品。
    # 消费品生产企业：生产消费品，支付工资，出售消费品换取利润，并根据供需信号调整生产计划。
    # 银行：所有个体的收入都会存入银行账户，银行会每年向消费者支付利息一次。
    # 政府：对工资收入和企业经营收入征税，并立即均分给消费者。
    # 你需要根据我的任务提示与我提供的数据图，给出你对一个仿真经济系统的调控方案。'''
    # response = client.chat.completions.create(
    #     model="glm-4v-plus",
    #     messages=[
    #     {
    #         "role": "user",
    #         "content": [{"type": "text", "text": system_prompt}],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image_url", "image_url": {"url": img2base64("/home/hhy/MEABM/价格.png")}},
    #             {"type": "text", "text": "这是经济仿真系统模拟产生的消费品价格曲线图"}
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image_url", "image_url": {"url": img2base64("/home/hhy/MEABM/产量.png")}},
    #             {"type": "text", "text": "这是经济仿真系统模拟产生的消费品产量曲线图"}
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image_url", "image_url": {"url": img2base64("/home/hhy/MEABM/失业率.png")}},
    #             {"type": "text", "text": "这是经济仿真系统模拟产生的失业率曲线图"}
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [{"type": "text", "text": 
    #                      '''你可以调控的经济指标有：存款利率，个人所得税率。你只能选择一个指标进行调控。
    #                      利率：增加50BP/降低50BP
    #                      税率：增加10BP/降低10BP
    #                      告诉我你的调控方案，以改善目前的经济状况。'''}]
    #     }
    #     ]
    # )
    # print(response.choices[0].message)

    # {
    #         "role": "assistant",
    #         "content": [
    #             {"text": '''这是一幅描绘自然风景的画，展现了一片宁静的湖泊，湖水呈现出深浅不一的蓝绿色调。
    #              湖边长着一棵盛开的樱花树，粉白色的花朵在阳光下显得格外鲜艳。树下有一位身穿蓝色衣服的人坐在船上，似乎正在享受这美好的时光。\n\n
    #              背景是连绵起伏的山脉，山体被绿色的植被覆盖，山顶则被云雾缭绕，增添了几分神秘感。
    #              远处还可以看到一座小亭子，它坐落在湖边的岩石上，与周围的自然环境和谐相融。\n\n
    #              整个画面给人一种宁静、和谐的感觉，仿佛置身于一个世外桃源之中。''', "type": "text"}
    #         ],
    #     },

# 根据你提供的图表信息，我们可以看到消费品价格、产量和失业率的变化趋势。为了改善目前的经济状况，我建议采取以下调控方案：
# **调控方案：降低存款利率50BP**
# 理由如下：
# 1. **消费品价格**：消费品价格呈上升趋势，这表明消费者对消费品的需求增加，导致价格上涨。通过降低存款利率，可以减少储蓄的吸引力，鼓励消费，从而缓解价格上涨的压力。
# 2. **消费品产量**：消费品产量波动较大，但总体上呈上升趋势。降低存款利率可以刺激消费需求，进一步推动消费品产量的增长，以满足市场需求。
# 3. **失业率**：失业率在初期较高，但随后逐渐下降。降低存款利率可以刺激消费需求，进而促进企业生产，增加就业机会，降低失业率。
# 综上所述，降低存款利率50BP可以帮助改善目前的经济状况，促进消费需求，增加就业机会，并缓解价格上涨的压力。