import os
from os import path


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tokenized.BPEtoken import BPE_tokenize_file
from tokenized.tokenizer import tokenize_string
from config import *

token_input_by_file = True

if token_input_by_file:
    sentences = LineSentence(os.path.join(tokenized_dir, 'tokens.txt')) 
else:
    with open(os.path.join(destination_dir, 'merged.txt'), 'r', encoding='utf-8') as file:
        string = file.read()
    token_ids,tokens = tokenize_string(string, tokenizer_path)
    print(tokens[:100])    

    sentences = []
    sentence = []
    for word in tokens:
        if word == '.':
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            sentence.append(word)
    if sentence:
        sentences.append(sentence)

print("sentences are ready.")


model = Word2Vec(vector_size=256, window=5, min_count=1, workers=4, sg=1)

model.build_vocab(sentences)

# 分步训练模型
for epoch in range(10):  # 假设训练 10 轮
    model.train(sentences, total_examples=model.corpus_count, epochs=1)
    print(f"Epoch {epoch + 1} completed")

# 保存模型
model.save("word2vec_sg.model")

# 获得词向量
word_vectors = model.wv

for word in word_vectors.index_to_key[:10]:
    print(f"Word: {word}, Vector: {word_vectors[word]}")
