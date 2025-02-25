import json
from typing import Dict, Tuple
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from adjustText import adjust_text


def load_medical_terms_with_categories(json_path: str) -> Tuple[set, Dict[str, str]]:
    """
    从JSON文件加载医疗术语及其类别，删除长度为 2 的单词
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            terms_data = json.load(file)

        medical_terms = set()
        term_categories = {}

        for term, info in terms_data.items():
            # 处理带有Ġ前缀的词
            clean_term = term[1:] if term.startswith('Ġ') else term
            clean_term = clean_term.lower()

            # 删除长度为 2 的词
            if len(clean_term) == 2:
                continue

            medical_terms.add(clean_term)
            term_categories[clean_term] = info['category']

        print(f"加载了 {len(medical_terms)} 个医疗相关词语")

        return medical_terms, term_categories

    except FileNotFoundError:
        print(f"找不到医疗术语文件: {json_path}")
        return set(), {}
    except Exception as e:
        print(f"加载医疗术语时出错: {str(e)}")
        return set(), {}


def visualize_medical_terms(term_categories: Dict[str, str], model_path: str):
    """
    使用t-SNE可视化医疗术语的词向量分布

    Args:
        term_categories: 词语类别映射
        model_path: Word2Vec模型路径
    """
    # 加载Word2Vec模型
    model = KeyedVectors.load(model_path)
    word2vec_model = model.wv

    # 收集词向量和对应的类别
    words = []
    categories = []
    vectors = []

    for word in term_categories.keys():
        # 处理可能的Ġ前缀
        word_variants = [word, word.lower(), word.upper(), 'Ġ' + word, 'Ġ' + word.lower(), 'Ġ' + word.upper()]

        # 尝试获取词向量
        for variant in word_variants:
            try:
                if variant in word2vec_model:
                    vector = word2vec_model[variant]
                    category = term_categories[word.lower()]

                    words.append(word)
                    categories.append(category)
                    vectors.append(vector)
                    break
            except KeyError:
                continue


    if not vectors:
        print("没有找到可用的词向量")
        return

    # 转换为numpy数组
    vectors = np.array(vectors)

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # 获取唯一的类别
    unique_categories = list(set(categories))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    category_color_dict = dict(zip(unique_categories, colors))

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 为每个类别绘制散点图
    texts = []
    for category in unique_categories:
        # 获取该类别的点
        mask = [c == category for c in categories]
        cat_vectors = reduced_vectors[mask]
        cat_words = [w for w, c in zip(words, categories) if c == category]

        # 绘制散点图
        plt.scatter(cat_vectors[:, 0], cat_vectors[:, 1],
                    c=[category_color_dict[category]],
                    label=f"{category} ({len(cat_words)})",
                    alpha=0.6)

        # 添加词汇标注
        for i, word in enumerate(cat_words):
            text = plt.text(
                cat_vectors[i, 0],
                cat_vectors[i, 1],
                word,
                fontsize=8,
                alpha=0.7
            )
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    plt.title("Medical Terms Distribution by Category")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 打印类别统计
    print("\n各类别的词语数量：")
    category_counts = Counter(categories)
    for category, count in category_counts.most_common():
        print(f"{category}: {count}")


def process_file(medical_terms_json_path: str, model_path: str):
    """处理文件并可视化医疗术语"""
    try:
        # 加载医疗术语及其类别
        medical_terms, term_categories = load_medical_terms_with_categories(medical_terms_json_path)

        if not medical_terms:
            return


        # 可视化
        visualize_medical_terms(term_categories, model_path)

        return term_categories

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    medical_terms_json_path = r"../data/medical_terms.json"
    model_path = "../representation/word2vec_model_10000.model"
    term_categories = process_file(medical_terms_json_path, model_path)