import os
import shutil
from os import path
from tqdm import tqdm
from transformers import GPT2TokenizerFast

#%% Configuration
file_path = 'D:\\studyyyy\\program\\nlp-winter\\data\\pdf_parsed\\merged.txt'

destination = 'D:\\studyyyy\\program\\nlp-winter\\data\\tokenized\\'

# writing to file or not
write_to_file = False

#%% code
# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print("Tokenizer initialized.")

# Read the file
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Tokenize the text
tokens = []
for line in tqdm(lines, desc="Tokenizing"):
    tokens.extend(tokenizer.tokenize(line, truncation=True, max_length=512))

# convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the first 100 tokens and token IDs
print(tokens[:100])
print(token_ids[:100])

# Write to file
if write_to_file:
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    
    with open(path.join(destination, 'BPE_tokenized.txt'), 'w', encoding='utf-8') as file:
        for token in tqdm(tokens, desc="Writing tokens"):
            file.write(token+ "\n")
    with open(path.join(destination, 'BPE_tokenized_ids.txt'), 'w', encoding='utf-8') as file:
        for token_id in tqdm(token_ids, desc="Writing token IDs"):
            file.write(str(token_id) + "\n")
