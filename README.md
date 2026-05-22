# 🇵🇰 Roman Urdu to English AI Translator

> A Groq-accelerated Neural Machine Translation (NMT) tool engineered to convert code-mixed, informal Roman Urdu into professional English.

 **Live Demo:** [Click Here to Use the App](https://huggingface.co/spaces/ugabooga/roman-urdu-translator)

---

## Re-Engineered Architecture
* **Inference Engine:** `Llama-3.1-8b-instant` via **Groq LPU** (Language Processing Unit) Infrastructure.
* **Prompt Strategy:** Few-Shot In-Context Learning with explicit role guardrails to isolate command verbs from translation targets.
* **Frontend UI:** Gradio (Python)
* **Hosting Pipeline:** Hugging Face Spaces

---

## Evaluation & Performance Metrics
To stress-test the translation pipeline beyond standard "golden-path" sentences, the system was sequentially evaluated against a curated **N=30 multi-complexity benchmark dataset** spanning casual dialogue, CS/ML technical jargon, heavy code-mixing, and localized idiomatic slang.

Linguistic accuracy was mathematically verified using the industry-standard `sacrebleu` library against natural human-written references.

### Production Results
* **True Average Inference Latency:** `0.1876 seconds` (Ultra-low latency powered by Groq LPUs)
* **BLEU Score:** `49.02` (High structural translation fidelity on messy, out-of-vocabulary real-world text)
* **chrF Score:** `64.78` (Character n-gram F-score validating robust handling of phonetic spelling inconsistencies)

### Performance Visualization
The complete benchmark breakdown is exported automatically to your local repository directory upon running the evaluation suite:

<img width="1089" height="489" alt="translator_performance" src="https://github.com/user-attachments/assets/c28c3e05-a4c5-4ebf-a2ce-22f816d0be0c" />

---

## How It Works
1. **Input Layer:** Captures raw, phonetically unstable Roman Urdu (e.g., *"Frontend aur backend ki API integration masla kar rahi hai."*)
2. **Context Guardrailing:** Injects a strict translation system instruction containing negative constraints to prevent the model from breaking its translation role when encountering English action words (e.g., *explain*, *fix*, *check*).
3. **Hardware Acceleration:** Dispatches token sequences to Groq's LPU hardware instances, securing near-instantaneous output.
4. **Deterministic Target Output:** Outputs polished English text (*"The frontend and backend API integration is causing problems."*)

---

## Local Installation & Benchmarking

### 1. Clone the Repository
```
git clone [https://github.com/GreatEpee/roman-urdu-nlp-translator.git]
cd roman-urdu-nlp-translator
```

### 2. Configure Environment & Dependencies
Initialize a virtual environment and install the verified dependency stack:
```
python -m venv venv
```

For Windows:
```
venv\Scripts\activate
```

For Mac/Linux:
```
source venv/bin/activate
```

Dependencies:
```
pip install -r requirements.txt
```

### 3. Setup Infrastructure Keys
Create a .env file in the root directory and append your access token:
```
GROQ_API_KEY=gsk_your_actual_groq_key_here
```

### 4. Execute System Components
To run the live interactive Gradio dashboard locally:
```
python app.py
```

To execute the production sequential testing harness and regenerate plots:
```
python evaluate_translator.py
```
