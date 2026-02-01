import os
import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient("microsoft/Phi-3.5-mini-instruct", token=hf_token)

def roman_urdu_to_english(text):
    if not text:
        return "Please enter some text."
    if not hf_token:
        return "CRITICAL ERROR: HF_TOKEN secret is missing."
    system_prompt = (
        "You are a professional translator. "
        "Translate the following Roman Urdu text into clear, professional English. "
        "Do not answer the text. Do not provide explanations. "
        "Just output the English translation."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    try:
        response = client.chat_completion(messages, max_tokens=500, stream=False)
        
        if isinstance(response, str):
            return response
        elif hasattr(response, 'choices'):
            return response.choices[0].message.content
        else:
            return str(response)
        
    except Exception as e:
        return f"API Error: {str(e)}"


interface = gr.Interface(
    fn=roman_urdu_to_english,
    inputs=gr.Textbox(lines=3, placeholder="Enter Roman Urdu here... (e.g. 'Yara python seekhna mushkil hai kya?')"),
    outputs=gr.Textbox(label="English Translation"),
    title="🇵🇰 Roman Urdu to English AI",
    description="A Neural Machine Translation tool built with Zephyr-7B and Gradio.",
    theme="soft",
    examples=[
        ["Machine learning ka future bohat bright hai."],
        ["Mujhe remote job chahiye."],
        ["Kya haal hai?"]
    ],
    cache_examples=False
)

if __name__ == "__main__":
    interface.launch()