import re
from collections import Counter
import matplotlib.pyplot as plt
import gc


def load_medical_terms(file_path: str) -> set:
    """从文件加载医疗术语"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            medical_terms = {line.strip().lower() for line in file if line.strip()}
        return medical_terms
    except FileNotFoundError:
        print(f"找不到医疗术语文件: {file_path}")
        return set()
    except Exception as e:
        print(f"加载医疗术语时出错: {str(e)}")
        return set()


def find_covid_pattern(text: str) -> list:
    """
    按 '.' 和 ',' 切分文本，并查找包含 'COVID-19' 相关词的句子
    """
    sentence_pattern = r'[^.,]+'
    sentences = re.findall(sentence_pattern, text)
    contexts = []

    for sentence in sentences:
        words = sentence.split()
        cleaned_words = [word.replace('Ġ', '').strip().lower() for word in words if word.strip()]

        if any(covid_variant in cleaned_words for covid_variant in ['covid', 'covid-19', 'covid19']):
            contexts.append(cleaned_words)

    return contexts


def analyze_medical_cooccurrence(contexts: list, medical_terms: set) -> dict:
    """分析医疗术语的共现频率"""
    medical_cooccurrence = []
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', '-', '19', 'covid', 'covid-19', 'covid19', 'from', 'positive', 'severe', 'clinical',
        'negative', 'other', 'time', 'high', 'related', 'associated', 'information', 'over', 'factor', 'specific'
    }

    for context in contexts:
        medical_words = [
            word for word in context
            if word in medical_terms and word not in stop_words
        ]
        medical_cooccurrence.extend(medical_words)

    return Counter(medical_cooccurrence)


def plot_medical_frequencies(word_freq: dict, top_n: int = 10):
    """绘制医疗术语频率统计图"""
    plt.figure(figsize=(12, 6))

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top_words:
        print("没有找到医疗术语")
        return

    words, freqs = zip(*top_words)

    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.title('Medical Terms Most Frequently Co-occurring with COVID-19')
    plt.xlabel('Medical Terms')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def process_file(tokens_file_path: str, medical_terms_path: str):
    """处理文件并分析医疗术语"""
    try:
        medical_terms = load_medical_terms(medical_terms_path)
        print(f"加载了 {len(medical_terms)} 个医疗相关词语")

        if not medical_terms:
            return

        with open(tokens_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        contexts = find_covid_pattern(text)

        if not contexts:
            print("未找到匹配的COVID-19模式")
            return

        medical_frequencies = analyze_medical_cooccurrence(contexts, medical_terms)
        gc.collect()
        if not medical_frequencies:
            print("未找到相关医疗术语")
            return

        print("\n与COVID-19最常共现的医疗术语:")
        for word, freq in sorted(medical_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{word}: {freq}")

        plot_medical_frequencies(medical_frequencies)

        return medical_frequencies

    except FileNotFoundError:
        print(f"找不到文件: {tokens_file_path}")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    tokens_file_path = r"H:\\document_parses\\document_parses\\data\\tokenized\\tokens.txt"
    medical_terms_path = r"C:\\Users\\HP\\Desktop\\pycharm\\put jupyter here\\winter school\\ICDS\\data\\entities_tokens.txt"
    medical_frequencies = process_file(tokens_file_path, medical_terms_path)
    gc.collect()