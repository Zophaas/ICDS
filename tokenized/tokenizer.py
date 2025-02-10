import os
from tokenizers import ByteLevelBPETokenizer
import multiprocessing
from tqdm import tqdm

def tokenize_chunk(chunk_args):
    string, tokenizer_path = chunk_args
    return  tokenize_string(string, tokenizer_path)

def tokenize_string(string, tokenizer_path):
    """Tokenize a file chunk"""
    tokenizer = ByteLevelBPETokenizer.from_file(os.path.join(tokenizer_path,'vocab.json'),
                                                os.path.join(tokenizer_path,'merges.txt'))
    encoded = tokenizer.encode(string)
    return encoded.ids, encoded.tokens

def write_tokenized_file(output_dir:str, input_file:str, vocab_dir:str, chunk_size = 10*1024*1024):
    # Prepare output file
    output_file = os.path.join(output_dir, 'tokenized_essays.txt')

    # Multiprocessing setup

    tokenizer_path = os.path.join(output_dir, 'tokenizer')
    with open(input_file, 'r', encoding='utf-8') as f, \
            open(output_file, 'w', encoding='utf-8') as out_f:
        pool = multiprocessing.Pool(processes=os.cpu_count())

        # Stream and tokenize chunks
        chunks = iter(lambda: f.read(chunk_size), '')
        chunk_args = [(chunk, os.path.join(vocab_dir, 'vocab.json')) for chunk in chunks]

        for tokenized_chunk in tqdm(pool.imap(tokenize_chunk, chunk_args)):
            out_f.write(' '.join(map(str, tokenized_chunk)) + '\n')

    print(f"Large file tokenization complete. Output in {output_file}")