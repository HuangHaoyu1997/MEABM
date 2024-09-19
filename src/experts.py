from config import Configuration
from api_key import api_key
import cv2, base64, requests

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
    
    config = Configuration()
    
    subfigs = split_img('./figs/bar-event-intervention.png')
    fig_names = [
        'price', 
        'interest rate ', 
        'employment rate', 
        'inflation rate', 
        'gini coefficient',
        'demand-supply imbalance', 
        'firm\'s capital and bank\'s assets', 
        'goods production', 
        'GDP', 
        'deposit per capita',
        'wage per capita',
        'tax revenue per capita',
        'standard deviation of workers\' wages',
        ]
    for fig_name, subfig in zip(fig_names, subfigs):
        prefix = f'This is the simulated curve of {fig_name} from the agent-based economic model.'
        response = get_expert_response(subfig, config)
        print(response)
    
    
    
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
    
    from prompts import user_prompt
    user_prompt.format(fig_name='price of goods',
                       now_step='400',
                       index_name='price',
                       event_name='economic crisis',
                       event_time='350',
                       event_effect='workers\' work will be reduced',)
    
    
