import sys
import os
import warnings
from contextlib import redirect_stdout
import difflib
import json
import os

# Path to your JSON file
file_path = "otherDirectory/x.py"  # Make sure this path points to your JSON content file (even if named .py)

# If the JSON content is embedded inside a Python file (not pure .json), extract it
def extract_json_from_py(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Try to extract the list structure (starting with [ and ending with ])
    start = content.find('[')
    end = content.rfind(']') + 1
    json_str = content[start:end]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("❌ Error decoding JSON:", e)
        return []

# Extract and join text
def combine_text(data):
    words = [entry["text"] for entry in data if "text" in entry]
    return " ".join(words)

data = extract_json_from_py(file_path)
if data:
    result = combine_text(data)
    print("✅ Combined Text:", result)
else:
    print("❌ Failed to load JSON or find text.")

# Suppress FutureWarnings (in case there are any real ones)
warnings.filterwarnings("ignore", category=FutureWarning)

with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
    import fasttext
    model = fasttext.load_model("lid.176.bin")

# ✅ Suppress FastText printed warnings (redirect stdout temporarily)

from langdetect import detect
import language_tool_python
#Cleaning The Hindi & Korean Words Text... Running This Only 1 Time Then Commenting Out
'''def clean_wordlist(input_file, output_file, allowed_script):
    import re

    def is_valid_word(word):
        if allowed_script == "hi":
            return re.fullmatch(r'[\u0900-\u097F]+', word) is not None
        elif allowed_script == "ko":
            return re.fullmatch(r'[\uAC00-\uD7A3]+', word) is not None
        else:
            return False

    words = set()

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            if is_valid_word(word):
                words.add(word)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for word in sorted(words):
            outfile.write(word + '\n')

    print(f"✅ Cleaned wordlist saved to: {output_file} ({len(words)} words)")
clean_wordlist("hi.txt", "hi_words_cleaned.txt", "hi")
clean_wordlist("ko.txt", "ko_words_cleaned.txt", "ko") '''
def load_wordlist(filepath):
    with open(filepath, encoding='utf-8') as f:
        return set(word.strip() for word in f.readlines())

def correct_word_custom(word, wordlist):
    matches = difflib.get_close_matches(word, wordlist, n=1, cutoff=0.8)
    if matches and matches[0] != word:
        print(f"❗ '{word}' → Suggested: '{matches[0]}'")
        return matches[0]
    return word

def correct_text_custom(text, wordlist):
    return ' '.join(correct_word_custom(w, wordlist) for w in text.split())


def correct_text_offline(text, language_code):
    try:
        if language_code=="en":
            tool = language_tool_python.LanguageTool('en-US')  # critical: don't use just 'en'
            matches = tool.check(text)

            if not matches:
                print("✅ No errors found.")
            else:
                print(f"🔍 {len(matches)} issues found.")

            corrected = language_tool_python.utils.correct(text, matches)
            return corrected
        elif language_code=="es":
            tool = language_tool_python.LanguageTool('es_AR')  # critical: don't use just 'en'
            matches = tool.check(text)

            if not matches:
                print("✅ No errors found.")
            else:
                print(f"🔍 {len(matches)} issues found.")

            corrected = language_tool_python.utils.correct(text, matches)
            return corrected
        elif language_code=="ar":
            tool = language_tool_python.LanguageTool('ar')

            matches = tool.check(text)

            # Display matches
            if matches:
                print(f"🔍 Found {len(matches)} issue(s):")
            else:
                print("✅ No issues found.")

            # Correct the sentence
            corrected = language_tool_python.utils.correct(text, matches)
            return corrected 
        elif language_code=="ru":
            tool = language_tool_python.LanguageTool('ru')  # 'ru' = Russian
            matches = tool.check(text)
            
            if matches:
                corrected = language_tool_python.utils.correct(text, matches)
                print("✅ Corrected:", corrected)
                print("🔍 Issues found:", len(matches))
                return corrected
            else:
                print("✅ No errors found.")
                return text
        elif language_code == "hi":
            wordlist = load_wordlist("hi_words_cleaned.txt")
            return correct_text_custom(text, wordlist)

        elif language_code == "ko":
            wordlist = load_wordlist("ko_words_cleaned.txt")
            return correct_text_custom(text, wordlist)
    except Exception as e:
            print(f"❌ Error: {e}")
            return text 

# Example
raw_text = "이것은 한국어 문장입니당 잘못됬죠?"
# Predict language
predictions = model.predict(raw_text)

# Output
label = predictions[0][0]  # Example: '__label__fr'
confidence = predictions[1][0]  # Example: 0.9998

# Clean the label
language_code = label.replace("__label__", "")
if language_code=="en":
    print("Detected Language From The Given Text is English")
elif language_code=="es":
    print("Detected Language From The Given Text is Spanish")
elif language_code=="hi":
    print("Detected Language From The Given Text is Hindi")
elif language_code=="ru":
    print("Detected Language From The Given Text is Russian")
elif language_code=="ko":
    print("Detected Language From The Given Text is Korean")
else:
    print("Detected Language From The Given Text is Arabic")
Correct_Text=correct_text_offline(raw_text, language_code)
print(f"Confidence: {confidence:.4f}")
print(Correct_Text)
print("OVER")

