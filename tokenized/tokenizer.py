import multiprocessing
import os
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice
import math

import config

merge = True

def tokenize_chunk(chunk_args):
    string, tokenizer_path = chunk_args
    ids, tokens = tokenize_string(string, tokenizer_path)
    return tokens

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
        pool = multiprocessing.Pool(processes=32)

        # Stream and tokenize chunks
        chunks = iter(lambda: f.read(chunk_size), '')
        chunk_args = [(chunk, vocab_dir) for chunk in chunks]

        for tokenized_chunk in tqdm(pool.imap(tokenize_chunk, chunk_args)):
            out_f.write(' '.join(map(str, tokenized_chunk)) + '\n')

    print(f"Large file tokenization complete. Output in {output_file}")


def process_chunk(args):
    """Process a chunk of files and save to a temporary output file"""
    chunk_files, destination_dir, tokenizer_path, tokenized_dir, chunk_id = args

    # Initialize tokenizer for this process
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tokenizer_path, 'vocab.json'),
        os.path.join(tokenizer_path, 'merges.txt')
    )

    # Create output file for this chunk
    output_path = os.path.join(tokenized_dir, f'tokenized_chunk_{chunk_id}.txt')
    with open(output_path, 'w') as output_file:
        for filename in tqdm(chunk_files):
            with open(os.path.join(destination_dir, filename), 'r', encoding='utf-8') as f:
                context = f.read()
            encoded = tokenizer.encode(context)
            output_file.write(' '.join(encoded.tokens) + '\n')

    return output_path


def merge_files(chunk_files, output_file):
    """Merge all chunk files into final output file"""
    with open(output_file, 'w') as outfile:
        for chunk_file in tqdm(chunk_files, desc="Merging files"):
            with open(chunk_file, 'r') as infile:
                outfile.write(infile.read())
            # Clean up chunk file
            os.remove(chunk_file)


def main():
    destination_dir = config.destination_dir
    tokenized_dir = config.tokenized_dir
    tokenizer_path = '../vocab'
    os.makedirs(tokenized_dir, exist_ok=True)

    # Get list of all files
    all_files = os.listdir(destination_dir)

    # Calculate chunk size based on number of CPU cores
    num_cores = 32  # This will return 32 on your system
    chunk_size = math.ceil(len(all_files) / num_cores)

    # Create chunks of files
    file_chunks = []
    for i in range(0, len(all_files), chunk_size):
        chunk = list(islice(all_files, i, i + chunk_size))
        file_chunks.append((chunk, destination_dir, tokenizer_path, tokenized_dir, i // chunk_size))

    # Process chunks in parallel with better progress tracking
    print(f"Processing {len(all_files)} files using {num_cores} cores...")
    chunk_files = []
    with mp.Pool(num_cores) as pool:
        results = pool.map_async(process_chunk, file_chunks)
        chunk_files = results.get()

    # Merge all chunk files into final output
    print("Merging chunk files...")
    merge_files(chunk_files, os.path.join(tokenized_dir, 'merged_tokenized.txt'))
    print("Done!")


if __name__ == '__main__':
    main()
