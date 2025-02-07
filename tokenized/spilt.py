import re
import os
from os import path
from tqdm import tqdm

#%% Configuration

file_path = 'D:\\studyyyy\\program\\nlp-winter\\data\\pdf_parsed\\merged.txt'

destination = 'D:\\studyyyy\\program\\nlp-winter\\data\\tokenized\\'

# Remove punctuation or not
remove_punctuation = True

# Write to file or not
write_to_file = False

#%% code

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

if remove_punctuation:
    words=re.split(r'\W+', content) 
else:
    words=content.split()

print(words[:50])

if write_to_file:
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    with open(os.path.join(destination,'spilt.txt'), "w", encoding="utf-8") as f:
        f.write("\n".join(words))