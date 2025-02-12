import os

from gensim.models import Word2Vec
import logging
from tqdm import tqdm

import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_text(input_file, max_sentences=50000):
    """
    Preprocess the input text file and return sentences for Word2Vec training.
    Each sentence is a list of tokens.

    Args:
        input_file: Path to input file
        max_sentences: Maximum number of sentences to process
    """
    sentences = []

    with open(input_file, 'r', encoding='utf-8') as f:
        # Read the single line and split by Ċ (sentence separator)
        text = f.read().strip()
        raw_sentences = text.split('Ċ')[:max_sentences]  # Cut at specified index

        for sentence in tqdm(raw_sentences):
            # Split the sentence into tokens
            # Handle special tokens (Ġ represents space before token)
            tokens = []
            current_token = ''

            for char in sentence.strip().split():
                tokens.append(char)

            # Add the last token if it exists
            if current_token:
                tokens.append(current_token.lower())

            # Only add sentences that have tokens
            if tokens:
                sentences.append(tokens)

    return sentences


def train_word2vec(sentences, vector_size=256, window=5, min_count=5):
    """
    Train a Word2Vec model on the preprocessed sentences.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum frequency of words to consider

    Returns:
        Trained Word2Vec model
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=32,
        sg=1,  # Skip-gram model
        compute_loss=True,
        negative=15,  # Increased negative sampling
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001,  # Final learning rate
        epochs=10,
    )

    return model


def main(input_file, output_model_path, max_sentences=50000):
    """
    Main function to process input file and train Word2Vec model.
    """
    # Process the input file
    print("Processing input file...")
    sentences = preprocess_text(input_file, max_sentences)

    print(f"Found {len(sentences)} sentences (max: {max_sentences})")
    print("Sample sentences:", sentences[:5])

    # Train the model
    print("\nTraining Word2Vec model using 32 CPU cores...")
    model = train_word2vec(sentences)

    # Print training loss if available
    if model.get_latest_training_loss() is not None:
        print(f"Final training loss: {model.get_latest_training_loss():.4f}")

    # Save the model
    model.save(output_model_path)
    print(f"\nModel saved to {output_model_path}")

    # Print some example similarities
    if len(model.wv.key_to_index) > 0:
        word = list(model.wv.key_to_index.keys())[0]
        print(f"\nExample similar words to '{word}':")
        similar_words = model.wv.most_similar(word, topn=5)
        for similar_word, score in similar_words:
            print(f"{similar_word}: {score:.4f}")


if __name__ == "__main__":
    input_file = os.path.join(config.tokenized_dir,'merged_tokenized.txt')
    output_model_path = "word2vec_model.model"
    main(input_file, output_model_path, max_sentences=50000)