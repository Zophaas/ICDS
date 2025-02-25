import json
import spacy

# 加载医学 NER 模型
nlp = spacy.load("en_ner_bc5cdr_md")

# 读取 JSON 词汇表
vocab_path = "../vocab/vocab.json"
output_path = "../data/medical_terms.json"

with open(vocab_path, "r", encoding="utf-8") as file:
    vocab = json.load(file)

# 处理医学相关术语
medical_terms = {}

for word, index in vocab.items():
    original_word = word  # 保存原始单词格式
    if word.startswith(("Ġ", "Ĳ", "ĳ", "Ċ")):  # 处理空格和特殊字符
        word = word[1:]
    
    doc = nlp(word)  # 用 spaCy 进行医学术语识别
    
    for ent in doc.ents:
        # 调试输出每个识别的实体及其类别
        print(f"Word: {word}, Entity: {ent.text}, Category: {ent.label_}")
        
        if ent.label_ in ["DISEASE", "CHEMICAL"]:
            # 记录原始词，索引以及类别
            medical_terms[original_word] = {
                "index": index,
                "category": ent.label_
            }

# 将结果写入 JSON 文件
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(medical_terms, json_file, ensure_ascii=False, indent=4)

print(f"已保存 {len(medical_terms)} 个医学术语到 {output_path}")

