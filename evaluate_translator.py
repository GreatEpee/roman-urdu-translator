import os
import time
import sacrebleu
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

test_data = [
    # --- Casual & Daily Conversation ---
    {"roman_urdu": "Khana kha liya tum ne?", "reference": "Did you eat?"},
    {"roman_urdu": "Aaj mausam kafi pyara hai, bahar chalain?", "reference": "The weather is quite lovely today, shall we go out?"},
    {"roman_urdu": "Mera wait karna, main 5 minute mein aa raha hoon.", "reference": "Wait for me, I am coming in 5 minutes."},
    {"roman_urdu": "Ghar farigh baith kar tang aa gaya hoon.", "reference": "I am tired of sitting idle at home."},
    {"roman_urdu": "Raste mein traffic bohat thi is liye late ho gaya.", "reference": "There was a lot of traffic on the way, which is why I am late."},
    {"roman_urdu": "Agar barish hui toh hum plan cancel kar dein ge.", "reference": "If it rains, we will cancel the plan."},
    {"roman_urdu": "Mujhe pata tha tum yahi kahoge.", "reference": "I knew you would say this."},
    {"roman_urdu": "Doctor ne dawai waqt par khane ka kaha hai.", "reference": "The doctor advised taking the medicine on time."},
    
    # --- Technical & Professional (CS/ML Focus) ---
    {"roman_urdu": "Code mein null pointer exception aa rahi hai, check karo.", "reference": "There is a null pointer exception in the code, please check it."},
    {"roman_urdu": "Client ne requirements change kar di hain last moment pe.", "reference": "The client changed the requirements at the last moment."},
    {"roman_urdu": "Model ki accuracy drop ho gayi hai naye data pe.", "reference": "The model's accuracy has dropped on the new data."},
    {"roman_urdu": "Server down hai, DevOps team ko tag karo slack pe.", "reference": "The server is down, tag the DevOps team on Slack."},
    {"roman_urdu": "Mujhe internship ki talash hai machine learning domain mein.", "reference": "I am looking for an internship in the machine learning domain."},
    {"roman_urdu": "Frontend aur backend ki API integration masla kar rahi hai.", "reference": "The API integration between the frontend and backend is causing issues."},
    {"roman_urdu": "Pull request review kar ke merge kar dena.", "reference": "Please review and merge the pull request."},
    
    # --- Heavy Code-Mixed (English/Urdu Hybrid) ---
    {"roman_urdu": "Bhai next week meeting schedule karni hai with the HR.", "reference": "Brother, we need to schedule a meeting with HR next week."},
    {"roman_urdu": "Mujhe ye logic samajh nahi aa rahi, thoda explain kar de.", "reference": "I don't understand this logic, please explain it a bit."},
    {"roman_urdu": "System crash kar gaya because of heavy load.", "reference": "The system crashed because of a heavy load."},
    {"roman_urdu": "Is feature ko jaldi push karo, deadline kal ki hai.", "reference": "Push this feature quickly, the deadline is tomorrow."},
    {"roman_urdu": "Bhai performance optimize karni paregi, UI bohat laggy hai.", "reference": "Brother, we will have to optimize the performance, the UI is very laggy."},
    {"roman_urdu": "Mujhe lagta hai yeh approach scalable nahi hai long term mein.", "reference": "I think this approach is not scalable in the long term."},
    
    # --- Slang & Idiomatic Expressions ---
    {"roman_urdu": "Aaj raat ka kya scene hai?", "reference": "What are the plans for tonight?"},
    {"roman_urdu": "Bohat khap hai is project mein, dimagh kharab ho gaya hai.", "reference": "There is too much hassle in this project, it's driving me crazy."},
    {"roman_urdu": "Chalo koi jugaad lagate hain is masle ka.", "reference": "Let's figure out a workaround for this problem."},
    {"roman_urdu": "Bhai topa kara diya usne end time pe.", "reference": "He bailed on me at the last minute."},
    {"roman_urdu": "Chuss maari hai tune, chup kar ja.", "reference": "You made a stupid joke, keep quiet."},
    {"roman_urdu": "Ye banda totally farigh hai.", "reference": "This guy is completely useless."},
    {"roman_urdu": "Kasam se, aisi baatein sun kar gussa aata hai.", "reference": "I swear, hearing such things makes me angry."},
    {"roman_urdu": "Yara usko samjhana deewar se sar phorne wali baat hai.", "reference": "Explaining it to him is like banging your head against a brick wall."},
    {"roman_urdu": "Baat ko lamba mat karo, point par aao.", "reference": "Don't drag this out, get straight to the point."}
]

def evaluate_pipeline():
    print("Booting up Sequential Groq NMT Evaluation...")
    
    predictions = []
    references = [[item["reference"] for item in test_data]]
    total_latency = 0

    for idx, item in enumerate(test_data):
        print(f"[{idx+1}/30] Translating: {item['roman_urdu']}")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['roman_urdu']}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        output_text = response.choices[0].message.content.strip()
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency
        predictions.append(output_text)
        
        print(f"Output: {output_text} | Latency: {latency:.2f}s\n")
        
        # 2.5sec sleep is there to fully clear Groq's 30 RPM limit and avoid 429 error
        if idx < len(test_data) - 1:
            time.sleep(2.5)

    avg_latency = total_latency / len(test_data)
    bleu = sacrebleu.corpus_bleu(predictions, references)
    chrf = sacrebleu.corpus_chrf(predictions, references)

    bleu_score = float(bleu.score)
    chrf_score = float(chrf.score)

    print("-" * 30)
    print("GROQ PRODUCTION EVALUATION RESULTS (N=30)")
    print(f"True Average Inference Latency: {avg_latency:.4f} seconds")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"chrF Score: {chrf_score:.2f}")
    print("-" * 30)


    print("\nGenerating performance visualization chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    metrics_names = ["BLEU Score", "chrF Score"]
    metrics_values = [bleu_score, chrf_score]
    if bleu_score > chrf_score:
        metrics_names = ["chrF Score", "BLEU Score"]
        metrics_values = [chrf_score, bleu_score]

    bars1 = ax1.bar(metrics_names, metrics_values, color=['#4CAF50', '#2196F3'], width=0.4)
    ax1.set_ylim(0, 110)
    ax1.set_title("Translation Quality Benchmark (Llama 3.1)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Score (0.0 to 100.0)", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
        
    bars2 = ax2.bar(["Avg Latency"], [avg_latency], color=['#FF9800'], width=0.2)
    ax2.set_ylim(0, max(2.0, avg_latency + 0.5))
    ax2.set_title("Inference Performance (Groq)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Time (Seconds)", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("translator_performance.png", bbox_inches='tight')
    print("Success! Charts saved smoothly as: translator_performance.png")

if __name__ == "__main__":
    evaluate_pipeline()