import re
from collections import Counter
import matplotlib.pyplot as plt
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


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


def chunk_file(file_path: str, chunk_size: int = 10 * 1024 * 1024) -> list:
    """将文件分割成块"""
    chunks = []
    current_chunk = []
    current_size = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file,desc = 'lines'):
            current_chunk.append(line)
            current_size += len(line.encode('utf-8'))

            if current_size >= chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:  # 添加最后一个块
            chunks.append(''.join(current_chunk))

    return chunks


def find_covid_pattern(text: str, medical_terms: set) -> Counter:
    """查找包含 'COVID-19' 相关词的句子并分析医疗术语"""

    # 按 . , 分割句子
    sentences = text.replace('\n', ' ').split('.')
    sentences = [s for sentence in sentences for s in sentence.split(',')]

    medical_cooccurrence = []

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', '-', '19', 'covid', 'covid-19', 'covid19', 'from', 'positive', 'severe', 'clinical',
        'negative', 'other', 'time', 'high', 'related', 'associated', 'information', 'over', 'factor', 'specific'
    }

    for sentence in tqdm(sentences):
        words = sentence.split()
        cleaned_words = [word.strip().lower() for word in words if word.strip()]

        if any(covid_variant in cleaned_words for covid_variant in ['covid', 'covid-19', 'covid19']):
            medical_words = [
                word for word in cleaned_words
                if word in medical_terms and word not in stop_words
            ]
            medical_cooccurrence.extend(medical_words)

    return Counter(medical_cooccurrence)


def process_chunk(chunk: str, medical_terms: set) -> Counter:
    """处理单个文本块"""
    return find_covid_pattern(chunk, medical_terms)


def plot_medical_frequencies(word_freq: dict, top_n: int = 10):
    """绘制医疗术语频率统计图"""
    plt.figure(figsize=(12, 6))

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top_words:
        print("没有找到医疗术语")
        return

    words, freqs = zip(*top_words)
    words = [word.replace('ġ','') for word in words]

    plt.bar(range(len(words)), freqs)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.title('Medical Terms Most Frequently Co-occurring with COVID-19')
    plt.xlabel('Medical Terms')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def process_file(tokens_file_path: str, medical_terms_path: str):
    """使用多进程处理文件并分析医疗术语"""
    try:
        # 加载医疗术语
        medical_terms = load_medical_terms(medical_terms_path)
        print(f"加载了 {len(medical_terms)} 个医疗相关词语")

        if not medical_terms:
            return

        # 将文件分块
        print("正在将文件分块...")
        chunks = chunk_file(tokens_file_path)
        print(f"文件被分成了 {len(chunks)} 个块")

        # 设置进程数
        num_processes = 32  # 留出一个CPU核心
        print(f"使用 {num_processes} 个进程进行处理")

        # 创建进程池并处理
        with Pool(num_processes) as pool:
            # 使用partial固定medical_terms参数
            process_func = partial(process_chunk, medical_terms=medical_terms)

            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(process_func, chunks),
                total=len(chunks),
                desc="处理进度"
            ))

        # 合并所有Counter结果
        medical_frequencies = Counter()
        for result in results:
            medical_frequencies.update(result)

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
    finally:
        gc.collect()


if __name__ == "__main__":
    tokens_file_path = '../cache/merged_tokenized.txt'
    medical_terms_path = '../data/entities_tokens.txt'
    medical_frequencies = process_file(tokens_file_path, medical_terms_path)
    gc.collect()
