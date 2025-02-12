from gensim.models import KeyedVectors, Word2Vec
import numpy as np


def load_medical_terms(file_path):
    """加载医疗相关词语"""
    medical_terms = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if term:  # 确保不是空行
                    medical_terms.add(term)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return set()
    return medical_terms


def find_medical_similarities(word2vec_model, target_word, medical_terms, topn=20):
    """找出与目标词最相似的医疗相关词语"""
    # 获取所有相似词
    try:
        all_similar = word2vec_model.wv.most_similar(target_word, topn=100)  # 获取更多词，以便筛选
    except KeyError:
        print(f"词 '{target_word}' 不在模型词表中")
        return []

    # 筛选医疗相关词
    medical_similar = []
    for word, similarity in all_similar:
        if word in medical_terms:
            medical_similar.append((word, similarity))
            if len(medical_similar) >= topn:
                break

    return medical_similar


# 主程序
def main():
    # 加载模型
    model_path = "../representation/word2vec_sg1.model"
    word2vec_model = Word2Vec.load(model_path)

    # 加载医疗词语
    medical_terms_path = r"C:\Users\HP\Desktop\pycharm\put jupyter here\winter school\ICDS\data\entities_tokens.txt"
    medical_terms = load_medical_terms(medical_terms_path)
    print(f"加载了 {len(medical_terms)} 个医疗相关词语")

    # 设置目标词
    target_word = "COVID"
    alternative_words = ["covid", "COVID", "ĠCOVID", "ĠCovid"]

    # 确保词在词表中
    if target_word not in word2vec_model.wv.key_to_index:
        print(f"'{target_word}' 不在词表中")
        for word in alternative_words:
            if word in word2vec_model.wv.key_to_index:
                target_word = word
                print(f"使用替代词 '{word}'")
                break
        else:
            print("未找到相关词")
            return

    # 获取相似的医疗词语
    similar_words = find_medical_similarities(word2vec_model, target_word, medical_terms)

    # 打印结果
    print(f"\n与 '{target_word}' 最相似的医疗相关词:")
    print("\n{:<20} {:<10}".format("词", "相似度"))
    print("-" * 30)
    for word, similarity in similar_words:
        print("{:<20} {:.4f}".format(word, similarity))


if __name__ == "__main__":
    main()