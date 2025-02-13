from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from adjustText import adjust_text
import gc

def process_and_visualize_embeddings(model_path, start_idx, end_idx, perplexity=5):
    """
    Load the model and visualize the word embeddings, keep the words with Ġ (remove the Ġ prefix) and filter out the words without Ġ

    model_path: str, path to the word2vec model

    start_idx: int, start index of the words to visualize

    end_idx: int, end index of the words to visualize

    perplexity: int, t-SNE perplexity parameter

    """
    # load the model
    model = KeyedVectors.load(model_path)
    word2vec_model = model.wv

    # Process the words: remove the Ġ prefix and keep the words with Ġ
    processed_words = []
    processed_vectors = []

    for word in word2vec_model.index_to_key[start_idx:end_idx]:
        if word.startswith('Ġ'):
            processed_word = word[1:]  # remove the Ġ prefix
            processed_words.append(processed_word)
            processed_vectors.append(word2vec_model[word])

    print(f"original word count: {end_idx - start_idx}")
    print(f"Number of words after processing: {len(processed_words)}")
    gc.collect()
    if not processed_words:
        print("No words with Ġ prefix found.")
        return [], []

    # Convert the processed vectors to a numpy array
    word_vectors = np.array(processed_vectors)

    # use t-SNE to reduce the dimensionality of the word vectors
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    gc.collect()

    # Visualize the word embeddings
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6)

    # Add text labels to the points
    texts = []
    for i, word in enumerate(processed_words):
        texts.append(plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1],
                              word, fontsize=8, alpha=0.7))

    # Adjust the text labels to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    plt.title("Word Embeddings Visualization (Processed Tokens)")
    plt.tight_layout()
    plt.show()

    return processed_words, reduced_vectors


# example usage
model_path = "D:\\studyyyy\\program\\nlp-git\\ICDS\\word2vec_sg1.model"
processed_words, vectors = process_and_visualize_embeddings(
    model_path=model_path,
    start_idx=300,
    end_idx=700,
    perplexity=5
)

# Print the first 10 processed words
print("\nexample processsed words:")
for word in processed_words[:10]:
    print(word)

gc.collect()