import config
import os
from tokenizers import ByteLevelBPETokenizer
import multiprocessing
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE


save_model = True

def train_tokenizer(input_files, vocab_dir, vocab_size=30000):
    """Train BPE tokenizer on large file"""
    tokenizer = ByteLevelBPETokenizer()
    # Train tokenizer first
    tokenizer.train(
        files=input_files,
        vocab_size=vocab_size,
        min_frequency=5,
        special_tokens=['<pad>', '<unk>', '<cls>', '<sep>']
    )
    tokenizer.save_model(vocab_dir)
    return tokenizer


def bpe_tokenization(input_file, output_dir, vocab_size=10000):
    """
    Perform high-performance BPE tokenization with parallel processing

    Args:
        input_file (str): Path to large input text file
        output_dir (str): Directory for output files
        vocab_size (int): Size of vocabulary
    """

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()

    # Create trainer
    trainer = BpeTrainer(
        special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'],
        vocab_size=vocab_size,
        min_frequency=2
    )

    # Train tokenizer
    tokenizer.train([input_file], trainer)

    if save_model:
        prompt = input('Sure to save model? [y/N]: ')
        if not prompt in ['Y','y']:
            return 0
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer
    tokenizer.save(os.path.join(output_dir, 'bpe_tokenizer.json'))

    return tokenizer


if __name__ == '__main__':
    # train_tokenizer(os.path.join(config.destination_dir,'merged.txt'), config.tokenized_dir, config.vocab_dir)
    file_list = [os.path.join(config.destination_dir,filename) for filename in os.listdir(config.destination_dir)]
    train_tokenizer(file_list,config.sample_dir)
