import os
import shutil
from os import path

from gensim.models import Word2Vec
from tokenized.BPEtoken import BPE_tokenize_file
from config import *

write_to_file = False

tokens,token_ids = BPE_tokenize_file(destination_dir, tokenized_dir, write_to_file)
model = Word2Vec(vector_size=256, window=5, min_count=1, workers=4, sg=1)

model.build_vocab(tokens)

# 分步训练模型
for epoch in range(10):  # 假设训练 10 轮
    model.train(tokens, total_examples=model.corpus_count, epochs=1)
    print(f"Epoch {epoch + 1} completed")

# 保存模型
model.save('word2vec_model.bin')

# 获得词向量
word_vectors = model.wv

#输出前 10 个词向量
for word in word_vectors.index_to_key[:10]:
    print(f"Word: {word}, Vector: {word_vectors[word]}")