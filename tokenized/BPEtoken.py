import os
import shutil
from os import path
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from config import *

write_to_file = False

def BPE_tokenize_file(input_file, destination, write_to_file):
    # Initialize the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print("Tokenizer initialized.")
    
    # Read the file
    with open(os.path.join(input_file, 'merged.txt'), 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Tokenize the text
    tokens = []
    for line in tqdm(lines, desc="Tokenizing"):
        tokens.extend(tokenizer.tokenize(line, truncation=True, max_length=512))
    
    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Write to file
    if write_to_file:
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)
        
        with open(path.join(destination, 'tokens.txt'), 'w', encoding='utf-8') as file:
            for token in tqdm(tokens, desc="Writing tokens"):
                file.write(token+ "\n")
        with open(path.join(destination, 'token_ids.txt'), 'w', encoding='utf-8') as file:
            for token_id in tqdm(token_ids, desc="Writing token IDs"):
                file.write(str(token_id) + "\n")

    return tokens, token_ids
    
if __name__ == "__main__":
    # Tokenize the file
    tokens,token_ids = BPE_tokenize_file(destination_dir, tokenized_dir, write_to_file)

    # Print the first 100 tokens and token IDs
    print(tokens[:100])
    print(token_ids[:100])