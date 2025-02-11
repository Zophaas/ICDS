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

model = Word2Vec.load("word2vec_sg.model")

word_list=['Ġto', 'Ġbe', 'Ġaffected', 'Ġby', 'ĠCO', 'VID', '-', '19', 'Ġwith', 'Ġboth']

word_vectors = {word: model.wv[word] for word in word_list if word in model.wv}

for word, vector in word_vectors.items():
    print(f"Word: {word}, Vector: {vector}")