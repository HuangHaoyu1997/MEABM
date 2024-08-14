import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Configuration
from api_key import api_key
import cv2, base64, requests

def get_expert_response(prefix:str, img, config:Configuration):
    
    success, encoded_image = cv2.imencode('.png', img)
    binary_image = encoded_image.tobytes()
    base64_image = base64.b64encode(binary_image).decode('utf-8')
    
    system_prompt = '''
    You are an expert economist with a strong background in economic theory and empirical analysis. 
    Your role is to analyze various economic data and figures from an agent-based economic model, providing insights and recommendations based on your findings.
    '''
    
    prompt = '''Give professional but brief analysis of the figure.'''
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": config.gpt_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": 
                [
                    {"type": "text", "text": prefix + prompt},
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
    from utils import split_img
    subfigs = split_img('./figs/log_step.png')
    response = get_expert_response('', subfigs[0], Configuration())
    print(response)