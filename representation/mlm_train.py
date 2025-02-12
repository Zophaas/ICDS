from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
from tqdm import tqdm
import gc
import re
import os
print(os.getcwd())


class ContextualWordRepresentations:
    def __init__(self, model_name='bert-base-uncased', batch_size=1):
        """
        Initialize the MLM model and tokenizer with CPU optimizations.
        Args:
            model_name: The name of the pretrained model to use
            batch_size: Size of batches for processing (smaller for CPU)
        """
        # Force CPU usage
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Set smaller batch size for CPU
        self.batch_size = batch_size

        # Load model with CPU memory optimizations
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model...")
        # Load model with CPU-specific optimizations
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,  # Optimize memory usage
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        self.model.to(self.device)
        self.model.eval()

        # Clear any unused memory
        gc.collect()
        torch.cuda.empty_cache()
        print('Finished loading!')

    def preprocess_text(self, input_file, max_sentences=50000):
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

    def get_contextual_embeddings(self, sentence, target_word=None):
        """
        Get contextual embeddings with memory-efficient processing.
        """
        # Free memory before processing
        gc.collect()

        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Process with torch.no_grad() to save memory
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]

        # Convert to numpy and clear torch tensors
        embeddings = hidden_states.squeeze(0).numpy()
        del hidden_states
        del outputs
        torch.cuda.empty_cache()

        # Get token to word mapping
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        if target_word:
            target_indices = [i for i, token in enumerate(tokens)
                              if target_word in token and not token.startswith('##')]
            if target_indices:
                return embeddings[target_indices[0]]
            return None

        return embeddings, tokens

    def process_file(self, input_file, output_file):
        """
        Process file in batches with memory management.
        """
        sentences = self.preprocess_text(input_file)
        print(f"Processing {len(sentences)} sentences...")

        # Dictionary to store word embeddings
        word_embeddings = {}

        # Process in batches
        for i in tqdm(range(0, len(sentences), self.batch_size)):
            batch = sentences[i:i + self.batch_size]

            for sentence in batch:
                try:
                    embeddings, tokens = self.get_contextual_embeddings(sentence)

                    # Store embeddings for each non-special token
                    for idx, token in enumerate(tokens):
                        if not token.startswith('[') and not token.startswith('##'):
                            if token not in word_embeddings:
                                word_embeddings[token] = []
                            word_embeddings[token].append(embeddings[idx])

                except RuntimeError as e:
                    print(f"Error processing sentence: {e}")
                    continue

            # Clear memory after each batch
            gc.collect()

        # Calculate average embeddings
        print("Calculating final embeddings...")
        averaged_embeddings = {}
        for word, embs in word_embeddings.items():
            averaged_embeddings[word] = np.mean(embs, axis=0)
            # Clear original embeddings to save memory
            del word_embeddings[word]
            if len(averaged_embeddings) % 1000 == 0:
                gc.collect()

        # Save embeddings
        np.save(output_file, averaged_embeddings)
        print(f"Saved embeddings for {len(averaged_embeddings)} words to {output_file}")

        return averaged_embeddings


def main():
    """
    Main function with CPU-specific settings.
    """
    input_file = "./cache/tokenized_100.txt"
    output_file = "contextual_embeddings.npy"

    # Initialize with CPU-specific settings
    print("Initializing model...")
    mlm = ContextualWordRepresentations()  # Small batch size for CPU

    # Process the file
    embeddings = mlm.process_file(input_file, output_file)

    # Clear final memory
    gc.collect()

    # Demonstrate usage
    print("\nDemonstrating contextual embeddings for a sample sentence:")
    sample_sentence = "The tibial deviation requires treatment"
    emb, tokens = mlm.get_contextual_embeddings(sample_sentence)
    print("Tokens:", tokens)
    print("Embedding shape for each token:", emb.shape)


if __name__ == "__main__":
    main()