from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from adjustText import adjust_text
import gc

def process_and_visualize_embeddings(model_path, start_idx=500, end_idx=1000, perplexity=5):
    """
    加载模型并可视化词嵌入，保留带Ġ的词（去掉Ġ前缀），过滤掉不带Ġ的词
    """
    # 加载模型
    model = KeyedVectors.load(model_path)
    word2vec_model = model.wv

    # 处理词表：保留带Ġ的词（去掉前缀），过滤掉不带Ġ的词
    processed_words = []
    processed_vectors = []

    for word in word2vec_model.index_to_key[start_idx:end_idx]:
        if word.startswith('Ġ'):
            # 保留带Ġ的词，但去掉Ġ前缀
            processed_word = word[1:]  # 去掉Ġ前缀
            processed_words.append(processed_word)
            processed_vectors.append(word2vec_model[word])

    print(f"原始词数: {end_idx - start_idx}")
    print(f"处理后词数: {len(processed_words)}")
    gc.collect()
    if not processed_words:
        print("没有找到带Ġ前缀的词")
        return [], []

    # 转换为numpy数组
    word_vectors = np.array(processed_vectors)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    gc.collect()

    # 创建可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6)

    # 添加词标签（使用处理后的词）
    texts = []
    for i, word in enumerate(processed_words):
        texts.append(plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1],
                              word, fontsize=8, alpha=0.7))

    # 自动调整文本位置避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    plt.title("Word Embeddings Visualization (Processed Tokens)")
    plt.tight_layout()
    plt.show()

    return processed_words, reduced_vectors


# 使用示例
model_path = "../representation/word2vec_sg1.model"
processed_words, vectors = process_and_visualize_embeddings(
    model_path=model_path,
    start_idx=500,
    end_idx=1000,
    perplexity=5
)

# 打印一些处理后的词示例
print("\n处理后的词示例:")
for word in processed_words[:10]:
    print(word)

gc.collect()