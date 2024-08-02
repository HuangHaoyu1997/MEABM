import requests
import cv2, base64
image = cv2.imread('bar-intervention.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_base64 = base64.b64encode(image_rgb.tobytes()).decode('utf-8')

# OpenAI API Key
api_key = "sk-proj-pkqVRU3aNsKQey6RZk0cT3BlbkFJF7O3g5nvhs0iv3akmYok"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What can you learn from this image?"
                    },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
    "max_tokens": 300
    }

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json()['choices'][0]['message']['content'])