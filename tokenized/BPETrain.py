import config
import os
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(input_file, output_dir, vocab_dir, vocab_size=5000):
    """Train BPE tokenizer on large file"""
    tokenizer = ByteLevelBPETokenizer()
    os.makedirs(output_dir, exist_ok=True)
    # Train tokenizer first
    tokenizer = train_tokenizer(input_file)
    tokenizer.save_model(vocab_dir)
    tokenizer.train(
        files=[input_file],
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=['<pad>', '<unk>', '<cls>', '<sep>']
    )
    return tokenizer

if __name__ == '__main__':
    train_tokenizer(os.path.join(config.destination_dir,'merged.txt'), config.tokenized_dir, config.vocab_dir)
