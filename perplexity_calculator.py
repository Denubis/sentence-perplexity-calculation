#!/usr/bin/env python
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
from pprint import pprint
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_tagged_content(text: str, tag_name: str) -> Optional[str]:
    """Extract content between specified XML-like tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def get_word_position(sentence: str, target_word: str) -> Optional[int]:
    """Get 1-based position of target word in sentence"""
    words = sentence.split()
    for i, word in enumerate(words, 1):
        if word.strip('.,!?;"\'') == target_word:
            return i
    return None

def get_word_before(sentence: str, target_word: str) -> Optional[str]:
    """Get word immediately preceding target word"""
    words = sentence.split()
    for i, word in enumerate(words):
        if word.strip('.,!?;"\'') == target_word and i > 0:
            return words[i-1]
    return None

def calculate_perplexities(sentences: List[str]) -> List[float]:
    """Calculate perplexity scores for a list of sentences"""
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(
        model_id='gpt2',
        add_start_token=False,
        predictions=sentences
    )
    return results["perplexities"]

def calculate_token_probabilities(sentence: str, target_word: str, model, tokenizer) -> float:
    """Calculate probability for target word given context"""
    # Put inputs on correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize input
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'].to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Find target word tokens
    target_tokens = tokenizer(target_word, add_special_tokens=False)['input_ids']
    
    # Find position of target word in sentence tokens
    sentence_tokens = tokenizer(sentence, add_special_tokens=False)['input_ids']
    
    # Tokenize target word with whitespace to match sentence tokenization
    target_with_space = " " + target_word  # Add leading space
    target_tokens = tokenizer(target_with_space, add_special_tokens=False)['input_ids']
    
    # Find starting position of target word in token sequence
    target_len = len(target_tokens)
    for i in range(len(sentence_tokens) - target_len + 1):
        if sentence_tokens[i:i+target_len] == target_tokens:
            start_pos = i
            break
    else:
        # Try without leading space as fallback
        target_tokens = tokenizer(target_word, add_special_tokens=False)['input_ids']
        target_len = len(target_tokens)
        for i in range(len(sentence_tokens) - target_len + 1):
            if sentence_tokens[i:i+target_len] == target_tokens:
                start_pos = i
                break
        else:
            raise ValueError(f"Could not find target word '{target_word}' in tokenized sentence")
    
    # Calculate probability for each token in target word
    target_probs = []
    for i, target_id in enumerate(target_tokens):
        pos = start_pos + i
        prob = torch.exp(log_probs[0, pos-1, target_id]).item()
        target_probs.append(prob)
    
    # Return geometric mean of probabilities for multi-token words
    return float(np.exp(np.mean(np.log(target_probs))))

def setup_model():
    """Initialize model and tokenizer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2-large"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def process_file(input_file: str) -> pd.DataFrame:
    """Process input CSV and generate output CSV"""
    df = pd.read_csv(input_file)
    
    results = []
    for _, row in df.iterrows():
        target_word = row['target_word']
        model_output = row['Model Output']
        model, tokenizer = setup_model()

        
        # Extract sentences
        pred_sentence = extract_tagged_content(model_output, "predictable_sentence")
        neut_sentence = extract_tagged_content(model_output, "neutral_sentence")
        
        if not pred_sentence or not neut_sentence:
            print(f"Warning: Could not extract sentences for {target_word}")
            continue
            
        # Get positions and preceding words
        position_pred = get_word_position(pred_sentence, target_word)
        position_neut = get_word_position(neut_sentence, target_word)
        
        if position_pred != position_neut:
            raise ValueError(f"Target word '{target_word}' appears in different positions: {position_pred} vs {position_neut}")
            
        position = position_pred  # Use either one since they're equal
        word_before_pred = get_word_before(pred_sentence, target_word)
        word_before_neut = get_word_before(neut_sentence, target_word)

        prob_pred = calculate_token_probabilities(pred_sentence, target_word, model, tokenizer)
        prob_neut = calculate_token_probabilities(neut_sentence, target_word, model, tokenizer)
        
        # Calculate perplexities
        perplexities = calculate_perplexities([pred_sentence, neut_sentence])
        # pprint(perplexities)

        results.append({
            'target_word': target_word,
            'position_in_sentence': position,
            'predictable_sentence': pred_sentence,  
            'neutral_sentence': neut_sentence,
            'word_before_target_predictable': word_before_pred,
            'word_before_target_neutral': word_before_neut,
            'sentence_perplexity_predictable': f"{perplexities[0]:.2f}",
            'sentence_perplexity_neutral': f"{perplexities[1]:.2f}",
            'target_word_probability_predictable': f"{prob_pred*100:.6f}%",
            'target_word_probability_neutral': f"{prob_neut*100:.6f}%"
        })
    
    return pd.DataFrame(results)


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main processing pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get all CSV files from data directory
    data_dir = Path('data')
    input_files = list(data_dir.glob('*.csv'))
    
    if not input_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    logger.info(f"Found {len(input_files)} CSV files to process")
    
    # Process each file and collect results
    all_results = []
    output = Path('output')
    output.mkdir(exist_ok=True)
    
    for input_file in tqdm(input_files, desc="Processing files"):
        try:
            logger.info(f"Processing {input_file}")
            results_df = process_file(input_file)
            
            # Add source file information
            results_df['source_file'] = input_file.name
            all_results.append(results_df)
            


            # Save individual file results
            output_file = output / f"processed_{input_file.name}"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        output = Path('output')
        output.mkdir(exist_ok=True)
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_output = output / "combined_results.csv"
        combined_df.to_csv(combined_output, index=False)
        logger.info(f"Saved combined results to {combined_output}")
        

if __name__ == "__main__":
    main()