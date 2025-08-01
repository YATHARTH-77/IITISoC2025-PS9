import requests
import json
import time
from typing import List, Optional

class MultiAPIGrammarCorrector:
    """
    Grammar correction system with automatic API key rotation for rate limit handling
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize with multiple API keys
        
        Args:
            api_keys (List[str]): List of Groq API keys
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.failed_keys = set()  # Track temporarily failed keys
        self.key_cooldown = {}  # Track when to retry failed keys
        self.cooldown_time = 300  # 5 minutes cooldown for rate-limited keys
        
    def get_current_api_key(self) -> Optional[str]:
        """
        Get the current active API key, skipping failed ones
        
        Returns:
            str: Current API key or None if all keys are failed
        """
        # Clean up expired cooldowns
        current_time = time.time()
        expired_keys = [key for key, cooldown_end in self.key_cooldown.items() 
                       if current_time > cooldown_end]
        
        for key in expired_keys:
            self.failed_keys.discard(self.api_keys[key])
            del self.key_cooldown[key]
        
        # Find next available key
        attempts = 0
        while attempts < len(self.api_keys):
            current_key = self.api_keys[self.current_key_index]
            
            if current_key not in self.failed_keys:
                return current_key
            
            # Move to next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
        
        return None  # All keys are currently failed
    
    def mark_key_as_failed(self, api_key: str, error_code: int):
        """
        Mark an API key as temporarily failed due to rate limiting
        
        Args:
            api_key (str): The failed API key
            error_code (int): HTTP error code
        """
        if error_code == 429:  # Rate limit exceeded
            self.failed_keys.add(api_key)
            key_index = self.api_keys.index(api_key)
            self.key_cooldown[key_index] = time.time() + self.cooldown_time
            print(f"⚠️ API Key {key_index + 1} rate limited. Switching to next key...")
    
    def rotate_to_next_key(self):
        """Move to the next API key in rotation"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
    
    def correct_text_with_groq(self, text: str, lang_name: str = "English", max_retries: int = 3) -> str:
        """
        Corrects spelling and grammar using Groq API with automatic key rotation
        
        Args:
            text (str): Input text requiring correction
            lang_name (str): Target language name
            max_retries (int): Maximum number of retry attempts
        
        Returns:
            str: Corrected text or error message
        """
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
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 300
        }
        
        for attempt in range(max_retries):
            current_key = self.get_current_api_key()
            
            if current_key is None:
                return "❌ All API keys are currently rate limited. Please try again later."
            
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(self.groq_endpoint, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"].strip()
                    key_num = self.current_key_index + 1
                    print(f"✅ Success with API Key {key_num}")
                    return result
                
                elif response.status_code == 429:  # Rate limit exceeded
                    self.mark_key_as_failed(current_key, 429)
                    if attempt < max_retries - 1:
                        self.rotate_to_next_key()
                        print(f"🔄 Retrying with next API key (Attempt {attempt + 2}/{max_retries})")
                        time.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        return f"❌ Rate limit exceeded on all available keys"
                
                elif response.status_code in [401, 403]:  # Authentication errors
                    print(f"❌ Authentication error with current key. Switching...")
                    self.rotate_to_next_key()
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return f"❌ Authentication failed: {response.status_code}"
                
                else:
                    return f"❌ Error {response.status_code}: {response.text}"
            
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error: {str(e)}")
                if attempt < max_retries - 1:
                    self.rotate_to_next_key()
                    time.sleep(2)  # Longer delay for network errors
                    continue
                else:
                    return f"❌ Network error: {str(e)}"
        
        return "❌ Maximum retry attempts exceeded"
    
    def get_key_status(self) -> dict:
        """
        Get status of all API keys
        
        Returns:
            dict: Status information for each key
        """
        status = {}
        current_time = time.time()
        
        for i, key in enumerate(self.api_keys):
            if key in self.failed_keys:
                if i in self.key_cooldown:
                    remaining_cooldown = max(0, self.key_cooldown[i] - current_time)
                    status[f"Key_{i+1}"] = f"Rate Limited (Cooldown: {remaining_cooldown:.0f}s)"
                else:
                    status[f"Key_{i+1}"] = "Failed"
            else:
                status[f"Key_{i+1}"] = "Active" if i == self.current_key_index else "Available"
        
        return status

# Configuration - Add your multiple API keys here
API_KEYS = [
    "#API Key",  # Primary key
    "your_second_api_key_here",  # Secondary key
    "your_third_api_key_here"    # Tertiary key
]

# Initialize the multi-API corrector
corrector = MultiAPIGrammarCorrector(API_KEYS)

# Test texts with grammar errors
texts = {
    "Hindi": "वह स्कूल जाता लेकिन वह पढ़ाई नहीं करता।",
    "German": "Ich gehen morgen zur Schule weil ich habe ein prüfung.",
    "Italian": "Lui andare a scuola ogni giorno ma lui non studia bene.",
    "French": "Je va à l'école chaque jour mais je pas faire mes devoirs.",
    "English": "She don't likes to play with they're toys.",
    "Turkish": "Ben yarın sinema gitmek istiyor.",
    "Korean": "나 학교 가 매일 하지만 공부 안해요.",
    "Russian": "Я хадил в magazin и купил hлеба и moloko",
    "Spanish": "Ella no gustar de jugar con sus amigos en el parque",
}

def run_correction_tests():
    """Run comprehensive tests with multiple API keys"""
    print("🚀 MULTI-API KEY GRAMMAR CORRECTION SYSTEM")
    print("=" * 55)
    
    # Show initial key status
    print("📊 Initial API Key Status:")
    for key, status in corrector.get_key_status().items():
        print(f"   {key}: {status}")
    print()
    
    # Process each text
    for lang, txt in texts.items():
        print(f"🌐 Language: {lang}")
        print(f"Original:  {txt}")
        
        corrected = corrector.correct_text_with_groq(txt, lang)
        print(f"Corrected: {corrected}")
        print("-" * 40)
    
    # Show final key status
    print("\n📊 Final API Key Status:")
    for key, status in corrector.get_key_status().items():
        print(f"   {key}: {status}")

# Main execution
if __name__ == "__main__":
    # Validate API keys are provided
    valid_keys = [key for key in API_KEYS if key and "your_" not in key]
    
    if len(valid_keys) < 1:
        print("❌ Please provide at least one valid API key in the API_KEYS list")
    else:
        print(f"✅ Initialized with {len(valid_keys)} API key(s)")
        
        # Update corrector with only valid keys
        corrector.api_keys = valid_keys
        
        # Run the tests
        run_correction_tests()
