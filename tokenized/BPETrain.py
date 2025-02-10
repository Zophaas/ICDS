import os
from tokenizers import ByteLevelBPETokenizer
from config import *
import os
from tokenizers import ByteLevelBPETokenizer
import multiprocessing


def train_tokenizer(input_file, vocab_size=5000):
    """Train BPE tokenizer on large file"""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=['<pad>', '<unk>', '<cls>', '<sep>']
    )
    return tokenizer


def tokenize_chunk(chunk_info):
    """Tokenize a file chunk"""
    chunk, tokenizer_path = chunk_info
    tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode(chunk)
    return encoded.ids


def large_file_bpe_tokenization(input_file, output_dir, vocab_dir, chunk_size=10 * 1024 * 1024):
    """
    Tokenize large files using memory-efficient streaming approach

    Args:
        input_file (str): Path to large input file
        output_dir (str): Directory to save tokenized files
        chunk_size (int): Size of file chunks to process
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train tokenizer first
    tokenizer = train_tokenizer(input_file)
    tokenizer_path = os.path.join(output_dir, 'tokenizer')
    tokenizer.save_model(vocab_dir)

'''
    # Prepare output file
    output_file = os.path.join(output_dir, 'tokenized_essays.txt')

    # Multiprocessing setup
    with open(input_file, 'r', encoding='utf-8') as f, \
            open(output_file, 'w', encoding='utf-8') as out_f:
        pool = multiprocessing.Pool(processes=os.cpu_count())

        # Stream and tokenize chunks
        chunks = iter(lambda: f.read(chunk_size), '')
        chunk_args = [(chunk, os.path.join(vocab_dir, 'vocab.json')) for chunk in chunks]

        for tokenized_chunk in pool.imap(tokenize_chunk, chunk_args):
            out_f.write(' '.join(map(str, tokenized_chunk)) + '\n')

    print(f"Large file tokenization complete. Output in {output_file}")
'''

large_file_bpe_tokenization(os.path.join(destination_dir,'merged.txt'), tokenized_dir, sample_dir)
