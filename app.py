import os
import gradio as gr
from dotenv import load_dotenv
from groq import Groq

# Load environment variables for local testing
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=groq_key)

def roman_urdu_to_english(text):
    if not text:
        return "Please enter some text."
    if not groq_key:
        return "CRITICAL ERROR: GROQ_API_KEY secret is missing."
        
    system_prompt = (
        "You are a professional translator. Translate the following Roman Urdu text into clear, professional English. "
        "Do not answer the text. Do not provide explanations. Just output the English translation.\n\n"
        "Here are some examples:\n"
        "Input: 'Yara python seekhna mushkil hai kya?'\n"
        "Output: 'Friend, is it difficult to learn Python?'\n"
        "Input: 'Bhai, code mein bug hai fix kar de.'\n"
        "Output: 'Brother, there is a bug in the code, please fix it.'\n"
        "Input: 'Mujhe remote job chahiye machine learning mein.'\n"
        "Output: 'I want a remote job in machine learning.'\n"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        output_text = response.choices[0].message.content.strip()
        if output_text:
            return output_text
        else:
            return "Error: Empty response received from the model."
            
    except Exception as e:
        return f"API Error: {str(e)}"


interface = gr.Interface(
    fn=roman_urdu_to_english,
    inputs=gr.Textbox(lines=3, placeholder="Enter Roman Urdu here... (e.g. 'Yara python seekhna mushkil hai kya?')"),
    outputs=gr.Textbox(label="English Translation"),
    title="🇵🇰 Roman Urdu to English AI",
    description="A Neural Machine Translation tool built with Llama 3.1 (Groq) and Gradio.",
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
