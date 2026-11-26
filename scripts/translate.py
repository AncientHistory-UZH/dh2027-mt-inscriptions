import pandas as pd
import os
import time
import json
from tqdm import tqdm
import concurrent.futures

from google import genai
from google.genai import types
import anthropic
import openai

MODELS = [
    "gemini-2.5-flash",   
    "claude-sonnet-4-0",
    "gpt-5-nano-2025-08-07"
]

LANGUAGES = ["DE", "EN", "FR", "IT", "ES"]

def translate_single_lang(text, target_lang, model):
    system_prompt = (
        f"You are an expert translator of Latin epigraphy. Translate the following Latin inscription into strictly {target_lang}. "
        "Maintain the historical accuracy. Preserve the meaning of brackets [] () where they indicate missing or supplied letters.\\n"
        "You MUST respond with ONLY the translation as plain text. Do not add conversational text or wrap in quotes. Make sure to ONLY translate."
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if "gemini" in model:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key: return "ERROR: NO KEY"
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1
                    )
                )
                return response.text.strip()
                
            elif "sonnet" in model or "claude" in model:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key: return "ERROR: NO KEY"
                client = anthropic.Anthropic(api_key=api_key, max_retries=0)
                response = client.messages.create(
                    model=model,
                    max_tokens=10000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.1
                )
                return response.content[0].text.strip()
                
            elif "gpt" in model:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key: return "ERROR: NO KEY"
                client = openai.OpenAI(api_key=api_key, max_retries=0)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ]
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg or "quota" in err_msg or "overloaded" in err_msg:
                time.sleep(1 + (2 ** attempt))
            else:
                return None
    return None

def translate_row(text, model):
    system_prompt = (
        "You are an expert translator of Latin epigraphy. Translate the following Latin inscription into ALL 5 requested languages simultaneously: German (DE), English (EN), French (FR), Italian (IT), and Spanish (ES). "
        "Maintain the historical accuracy. Preserve the meaning of brackets [] () where they indicate missing or supplied letters.\\n"
        "You MUST respond with ONLY a valid, raw JSON object containing the translations. Do not wrap in markdown or add conversational text. Make sure to ONLY translate.\\n"
        'Expected format: {"DE": "...", "EN": "...", "FR": "...", "IT": "...", "ES": "..."}'
    )
    
    # Try fast batch JSON first
    max_retries = 2
    for attempt in range(max_retries):
        try:
            raw_output = ""
            if "gemini" in model:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key: return {lang: "ERROR: NO KEY" for lang in LANGUAGES}
                
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                raw_output = response.text.strip()
                
            elif "sonnet" in model or "claude" in model:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key: return {lang: "ERROR: NO KEY" for lang in LANGUAGES}
                
                client = anthropic.Anthropic(api_key=api_key, max_retries=0)
                response = client.messages.create(
                    model=model,
                    max_tokens=10000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "{"}
                    ],
                    temperature=0.1
                )
                raw_output = "{" + response.content[0].text.strip()
                
            elif "gpt" in model:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key: return {lang: "ERROR: NO KEY" for lang in LANGUAGES}
                
                client = openai.OpenAI(api_key=api_key, max_retries=0)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    # Remove temperature specifically for OpenAI o-series/nano models 
                    response_format={"type": "json_object"}
                )
                raw_output = response.choices[0].message.content.strip()
            
            # Clean possible markdown
            if raw_output.startswith("```"):
                lines = raw_output.split("\\n")
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                raw_output = "\\n".join(lines).strip()
            
            parsed = json.loads(raw_output)
            
            # --- CONTROL MECHANISM ---
            # Ensure the output is a dictionary.
            if not isinstance(parsed, dict):
                raise ValueError("Output is not a JSON dictionary")
            
            # Check for missing keys or invalid non-string outputs
            missing_keys = [lang for lang in LANGUAGES if lang not in parsed or not isinstance(parsed[lang], str)]
            if missing_keys:
                raise ValueError(f"Missing or invalid language translations: {missing_keys}")
                
            return parsed
            
        except (json.JSONDecodeError, ValueError) as ve:
            # Control mechanism caught a truncation/bad output
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                # BREAK and fallback to individual isolated requests!
                break
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg or "quota" in err_msg or "overloaded" in err_msg:
                if attempt < max_retries - 1:
                    time.sleep(1 + (2 ** attempt))
                    continue
                else:
                    return {lang: None for lang in LANGUAGES}
            else:
                print(f"\\nFATAL API Error with {model}: {e}")
                return {lang: None for lang in LANGUAGES}
                
    # --- DYNAMIC FALLBACK ---
    # If the JSON loop truncated on a massive text, use exact singles.
    # print(f"\\n[Fallback] Deep text bypass for {model}...")
    
    fallback_results = {}
    for lang in LANGUAGES:
        fallback_results[lang] = translate_single_lang(text, lang, model)
        if "gemini" in model:
            time.sleep(1) # Strict safety padding
            
    return fallback_results


def run_translations(input_csv, output_dir, limit=None):
    df = pd.read_csv(input_csv)
    if limit: df = df.head(limit)
    os.makedirs(output_dir, exist_ok=True)
    
    for model in MODELS:
        print(f"\\n--- Translating with {model} ---")
        results = []
        
        # Google heavily rate-limits free tier to 15 RPM. 
        workers = 1 if "gemini" in model else 5
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # 1 target = 1 row because 1 row returns 5 languages!
            futures = []
            for idx, row in df.iterrows():
                res = {"inschrift_id": row['inschrift_id'], "latin_text": row['text']}
                results.append(res)
                futures.append((res, executor.submit(translate_row, row['text'], model)))
            
            with tqdm(total=len(futures), desc=f"Translations ({model})") as pbar:
                for res, future in futures:
                    translated_dict = future.result()
                    # Populate the res dictionary
                    for lang in LANGUAGES:
                        res[lang] = translated_dict.get(lang) if isinstance(translated_dict, dict) else None
                    pbar.update(1)
                    
        # Change output format to JSON so commas in inscriptions don't screw up standard CSV readers
        # that aren't strict RFC 4180 compliant.
        with open(os.path.join(output_dir, f"translations_{model}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"\\nSaved {model} translations to JSON.")

if __name__ == "__main__":
    run_translations("data/filtered_inscriptions.csv", "data/translations", limit=10)
