import pandas as pd
import re
import os

def is_greek(text):
    if not isinstance(text, str):
        return False
    if text.strip() in ['"""GR"""', '"GR"', 'GR']:
        return True
    
    # Count greek letters
    greek_pattern = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
    greek_chars = len(greek_pattern.findall(text))
    # Count total letters
    total_letters = len([c for c in text if c.isalpha()])
    
    if total_letters == 0:
        return False
    
    # If more than 30% of the alphabetical characters are Greek, classify as Greek
    if greek_chars / total_letters > 0.3:
        return True
        
    return False

def clean_for_length(text):
    if not isinstance(text, str):
        return ""
    # Remove common editorial marks and brackets
    clean_text = re.sub(r'[\[\]\(\)\{\}\<\>⟦⟧]', '', text)
    return clean_text.strip()

def preprocess(input_path, output_path, min_length=50):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # Filter out missing text
    df = df.dropna(subset=['text'])
    print(f"Total rows: {len(df)}")
    
    # 1. Filter out Greek inscriptions
    df['is_greek'] = df['text'].apply(is_greek)
    filtered_df = df[~df['is_greek']].copy()
    print(f"Rows after removing Greek: {len(filtered_df)}")
    
    # 2. Filter by minimum character length after removing brackets
    filtered_df['clean_length'] = filtered_df['text'].apply(lambda x: len(clean_for_length(x)))
    final_df = filtered_df[filtered_df['clean_length'] >= min_length].copy()
    
    # Dropping intermediate columns
    final_df = final_df.drop(columns=['is_greek', 'clean_length'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Processed resulting rows: {len(final_df)}. Kept rows after filtering Greek and length >= {min_length}.")

if __name__ == "__main__":
    preprocess("data/edcs-id-text.csv", "data/filtered_inscriptions.csv", min_length=50)
