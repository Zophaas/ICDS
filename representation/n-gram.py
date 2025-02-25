from ngram_vectorizer import NgramVectorizer


def main():
    # 获取tokens
    with open('/root/autodl-tmp/document_parses/tokenized/merged_tokenized.txt') as f:
        tokens = f.read().split(' ')

    # 初始化N-gram向量化器
    vectorizer = NgramVectorizer(n=3, vector_size=256)

    # 训练模型（使用10个epochs）
    vectorizer.fit(tokens, epochs=1)

    # 保存词向量
    vectorizer.save_vectors('ngram_vectors.txt')

    # 示例：打印前几个词的向量
    print("\nExample word vectors:")
    for token in tokens[:5]:
        vector = vectorizer.get_vector(token)
        print(f"{token}: {vector[:5]}...")  # 只打印向量的前5个值


if __name__ == "__main__":
    main()