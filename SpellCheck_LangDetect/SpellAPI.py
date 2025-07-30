import requests
import json

GROQ_API_KEY = "#API KEY"  # 🔒 Replace this
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def correct_text_with_groq(text, lang_name="English"):
    """
    Corrects spelling and grammar using Groq LLaMA3 model in the specified language.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # System-level instruction to stay in the correct language
    system_prompt = (
        f"You are a professional language editor for {lang_name}. "
        f"Only correct spelling and grammar in {lang_name}. "
        f"Do not rewrite, rephrase, or remove any words."
    )

    # User message should include the actual input text
    user_prompt = (
        f"Correct only the grammar and spelling in the following {lang_name} text.\n\n"
        f"❗ Do not REMOVE, skip, change the position of, or rewrite any word under any condition STRICTLY.\n"
        f"❗ Do not translate or simplify the text.\n"
        f"❗ Return only the corrected version of the text, without explanation or extra comments STRICTLY.\n\n"
        f"Text:\n{text}\n\nCorrected:"
    )


    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ Error {response.status_code}: {response.text}"



# 🧪 Sample usage
texts = {
    "German": "Ich gehen morgen zur Schule weil ich habe ein prüfung.",
    "Italian": "Lui andare a scuola ogni giorno ma lui non studia bene.",
    "French": "Je aller au marché et acheter des pomme et du lait."
}


for lang, txt in texts.items():
    corrected = correct_text_with_groq(txt, lang)
    print("✅ Corrected:", corrected)
