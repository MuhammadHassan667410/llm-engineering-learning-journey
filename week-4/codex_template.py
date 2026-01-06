import os
import requests
import json
from openai import OpenAI

def run_codex(prompt, model="gpt-5.1-codex", api_key=None):
    """
    A simplified, unified function to call OpenAI Codex models.
    
    Handles:
    1. New 'Responses API' models (e.g., gpt-5.1-codex)
    2. Standard Chat models (e.g., gpt-5.1-chat-latest, gpt-4o)
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("API Key is missing. Set OPENAI_API_KEY env var or pass it explicitly.")

    print(f"Generating code using {model}...")

    # --- Strategy 1: New Responses API (for gpt-5.1-codex) ---
    if "codex" in model and "gpt-5" in model:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        data = {
            "model": model,
            "instructions": "You are an expert coding assistant. Return only the code.",
            "input": prompt,
            "max_output_tokens": 2000
        }
        
        try:
            resp = requests.post(url, headers=headers, json=data)
            if resp.status_code == 200:
                result = resp.json()
                # Extract text from the structured response
                output_text = []
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                output_text.append(block.get("text"))
                return "\n".join(output_text)
            else:
                return f"Error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Request failed: {str(e)}"

    # --- Strategy 2: Standard Chat API (for all other models) ---
    else:
        client = OpenAI(api_key=key)
        try:
            # Note: GPT-5/o1 models use 'max_completion_tokens'
            # Older GPT-4 models use 'max_tokens'
            param_name = "max_completion_tokens" if "gpt-5" in model or "o1" in model else "max_tokens"
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **{param_name: 2000}
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Chat API Error: {str(e)}"

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Use the new specialized Codex model
    code = run_codex("Write a Python function to check for prime numbers.")
    print("\n--- Output from gpt-5.1-codex ---")
    print(code)

    # 2. Use a standard model easily with the same function
    # code_chat = run_codex("Explain this code", model="gpt-5.1-chat-latest")
    # print(code_chat)
