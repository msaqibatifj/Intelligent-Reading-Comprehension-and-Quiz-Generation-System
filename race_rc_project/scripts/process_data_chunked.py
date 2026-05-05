"""Process raw RACE data in 5k-row chunks for Model A and Model B.

This script:
- Loads raw CSV files (train, test, validation)
- Processes in 5k-row chunks to avoid memory issues
- Extracts features for Model A (Q&A verification) and Model B (distractor/hint)
- Saves processed data to data/processed/

Usage:
    python race_rc_project/scripts/process_data_chunked.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from preprocessing import FeatureEngineer, prepare_qa_dataset, prepare_distractor_dataset


DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
CHUNK_SIZE = 5000


def find_raw_files():
    """Find all raw CSV files in data/raw/ (recursively)."""
    files = list(DATA_RAW.rglob('*.csv'))
    # Also check top-level test.csv if it exists
    if (DATA_RAW / 'test.csv').exists():
        files.append(DATA_RAW / 'test.csv')
    return sorted(set(files))


def load_options_safely(options_str):
    """Safely parse options from CSV (may be pipe-delimited or list-like)."""
    if isinstance(options_str, list):
        return options_str
    if isinstance(options_str, str):
        # Try pipe-separated first
        if '|' in options_str:
            return [o.strip() for o in options_str.split('|')]
        # Try bracket notation
        if options_str.startswith('[') and options_str.endswith(']'):
            try:
                return eval(options_str)
            except:
                pass
    return [str(options_str)]


def normalize_dataframe(df):
    """Ensure dataframe has required columns with safe parsing."""
    required = ['article', 'question', 'options', 'answer']
    
    # Rename 'passage' to 'article' if needed
    if 'passage' in df.columns and 'article' not in df.columns:
        df = df.rename(columns={'passage': 'article'})
    
    # Ensure all required columns exist
    for col in required:
        if col not in df.columns:
            if col == 'options':
                df[col] = [['Option A', 'Option B', 'Option C', 'Option D']] * len(df)
            elif col == 'answer':
                df[col] = ['A'] * len(df)
            else:
                df[col] = [''] * len(df)
    
    # Parse options
    df['options'] = df['options'].apply(load_options_safely)
    
    return df[required]


def process_chunk(df_chunk, chunk_id, feature_engineer, fit_fe=False):
    """Process a single chunk of data."""
    # Prepare QA dataset (for Model A)
    qa_data = []
    for idx, row in df_chunk.iterrows():
        article = row.get('article', '')
        question = row.get('question', '')
        options = row.get('options', [])
        answer = row.get('answer', 'A')
        
        if isinstance(options, str):
            options = load_options_safely(options)
        
        if not isinstance(options, list) or len(options) == 0:
            options = ['Option A', 'Option B', 'Option C', 'Option D']
        
        # Convert answer letter to index
        if isinstance(answer, str) and len(answer) == 1 and answer.upper() in ['A','B','C','D']:
            ans_idx = ord(answer.upper()) - ord('A')
        else:
            ans_idx = 0
        
        if ans_idx >= len(options):
            ans_idx = 0
        
        correct_answer = options[ans_idx]
        
        # Create a row for each option (Model A format)
        for opt_idx, option in enumerate(options):
            label = 1 if opt_idx == ans_idx else 0
            qa_data.append({
                'article': article,
                'question': question,
                'option': option,
                'label': label,
                'answer': correct_answer
            })
    
    # Prepare distractor dataset (for Model B)
    distractor_data = []
    for idx, row in df_chunk.iterrows():
        article = row.get('article', '')
        question = row.get('question', '')
        options = row.get('options', [])
        answer = row.get('answer', 'A')
        
        if isinstance(options, str):
            options = load_options_safely(options)
        
        if not isinstance(options, list) or len(options) == 0:
            options = ['Option A', 'Option B', 'Option C', 'Option D']
        
        if isinstance(answer, str) and len(answer) == 1 and answer.upper() in ['A','B','C','D']:
            ans_idx = ord(answer.upper()) - ord('A')
        else:
            ans_idx = 0
        
        if ans_idx >= len(options):
            ans_idx = 0
        
        correct_answer = options[ans_idx]
        distractors = [opt for i, opt in enumerate(options) if i != ans_idx]
        
        distractor_data.append({
            'article': article,
            'question': question,
            'correct_answer': correct_answer,
            'distractors': str(distractors)
        })
    
    df_qa = pd.DataFrame(qa_data)
    df_dist = pd.DataFrame(distractor_data)
    
    return df_qa, df_dist


def process_file(csv_path):
    """Process a single raw CSV file in chunks."""
    print(f"\nProcessing {csv_path}...")
    
    df = pd.read_csv(csv_path)
    df = normalize_dataframe(df)
    
    print(f"  Total rows: {len(df)}")
    
    feature_engineer = FeatureEngineer(max_features=5000)
    qa_chunks = []
    dist_chunks = []
    
    num_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(df))
        df_chunk = df.iloc[start_idx:end_idx]
        
        print(f"  Processing chunk {chunk_id + 1}/{num_chunks} (rows {start_idx}-{end_idx})...")
        
        try:
            df_qa, df_dist = process_chunk(
                df_chunk, chunk_id, feature_engineer,
                fit_fe=(chunk_id == 0)  # Fit on first chunk only
            )
            qa_chunks.append(df_qa)
            dist_chunks.append(df_dist)
        except Exception as e:
            print(f"    Error processing chunk {chunk_id}: {e}")
            traceback.print_exc()
    
    # Combine all chunks
    if qa_chunks:
        df_qa_all = pd.concat(qa_chunks, ignore_index=True)
        df_dist_all = pd.concat(dist_chunks, ignore_index=True)
        
        # Save processed data
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        
        # Derive output filename from input
        stem = csv_path.stem
        qa_out = DATA_PROCESSED / f'{stem}_qa.csv'
        dist_out = DATA_PROCESSED / f'{stem}_distractor.csv'
        
        df_qa_all.to_csv(qa_out, index=False)
        df_dist_all.to_csv(dist_out, index=False)
        
        print(f"  Saved {qa_out} ({len(df_qa_all)} rows)")
        print(f"  Saved {dist_out} ({len(df_dist_all)} rows)")


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    csv_files = find_raw_files()
    if not csv_files:
        print("No raw CSV files found in data/raw/")
        return
    
    print(f"Found {len(csv_files)} raw CSV files to process:")
    for f in csv_files:
        print(f"  - {f}")
    
    for csv_path in csv_files:
        process_file(csv_path)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
