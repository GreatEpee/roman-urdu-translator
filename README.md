# 🇵🇰 Roman Urdu to English AI Translator

> A Neural Machine Translation (NMT) tool designed to translate informal Roman Urdu into professional English.

 **Live Demo:** [Click Here to Use the App](https://ugabooga-roman-urdu-translator.hf.space/?logs=container&__theme=dark&deep_link=CYKdbXmqqog)

## Architecture
- **Model:** Qwen 2.5-72B-Instruct (State-of-the-art Open Source LLM)
- **Frontend:** Gradio (Python)
- **Deployment:** Hugging Face Spaces (Serverless CPU)
- **API:** Hugging Face Inference API

## How It Works
1. **Input:** User enters Roman Urdu (e.g., *"Yara python seekhna mushkil hai kya?"*)
2. **Prompt Engineering:** The app injects a system prompt to enforce "Professional Translator" behavior.
3. **Inference:** The Qwen model processes the tokenized input and predicts the English translation.
4. **Output:** Clean English text (e.g., *"Friend, is it difficult to learn Python?"*)
<img width="1748" height="634" alt="Screenshot 2026-02-01 115606" src="https://github.com/user-attachments/assets/d626fef2-2cb6-49b5-8b46-fc2efdbfbda8" />
<img width="1686" height="570" alt="Screenshot 2026-02-01 115448" src="https://github.com/user-attachments/assets/e5ef92a4-d48d-43f1-adb1-2988e0467525" />

## Local Installation
1. Clone the repo:
   ```
   git clone https://github.com/GreatEpee/roman-urdu-nlp-translator.git
   ```
2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your API Key:
   ```
   # Linux/Mac
   export HF_TOKEN="your_huggingface_key_here"

   # Windows (PowerShell)
   $env:HF_TOKEN="your_huggingface_key_here"
   ```
4. Run the app:
   ```
   python app.py
   ```
