import random
import re

from tqdm import tqdm
import json
import os
import shutil
from config import *

#%% Configuration

random_start = False   # Start from a random paper, DEBUGGING ONLY!

cutoff = -1   # How many papers to run on, -1 for all, disable random start for large numbers

merge = False    # Merge the file into one or not

copy_merge_to_wd = False  # Leave as False

copy_samples = False  # Leave as False

#%% Code
if random_start:
    start_from = random.randint(1,10000)
else:
    start_from = 0

if copy_merge_to_wd and (cutoff>1000):
    prompt = input('File may be too large, are you sure to copy? [y/N]')
    if not prompt in ('y', 'Y'):
        copy_merge_to_wd = False

if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
os.makedirs(destination_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)


if merge:
    fm = open(os.path.join(destination_dir, 'merged.txt'), 'w', encoding = 'utf-8')

for filename in tqdm(os.listdir(base_dir)[start_from:start_from + cutoff]):
    with open(os.path.join(base_dir, filename), 'r', encoding = 'utf-8') as f:
        json_data = json.load(f)
        paper_id = json_data['paper_id']
        title = json_data['metadata']['title']
        abstract = json_data['abstract'][0]['text'] if json_data['abstract'] else ''
        body_text_raw = json_data['body_text']
        body_text = ''
        for segment in body_text_raw:
            raw_text = segment['text']
            for cite in segment['cite_spans']:
                if cite['ref_id']!='':
                    raw_text = raw_text.replace(cite['text'], '')
            body_text+= raw_text.strip() + '\n'
        body_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', body_text)
        body_text_purged = re.sub(r'\[\d+]', '', body_text)
        # print(cite['text'],end=' ') if not random.randint(0,10) else None

    if not merge:
        with open(os.path.join(destination_dir, paper_id + '.txt'), 'w', encoding = 'utf-8') as f:
            f.write(title+'\n')
            f.write(abstract+'\n')
            f.write(body_text_purged + '\n')

    else:
        fm.write(paper_id+'\n')
        fm.write(title+'\n')
        fm.write(abstract+'\n')
        fm.write(body_text_purged + '\n')

if merge:
    fm.close()


if copy_merge_to_wd:
    shutil.copy(os.path.join(destination_dir, 'merged.txt'), os.path.join(sample_dir, 'merged.txt'))

if copy_samples:
    parsed_files = os.listdir(destination_dir)
    random.shuffle(parsed_files)
    for f in parsed_files[:10]:
        shutil.copy(os.path.join(destination_dir, f), os.path.join(sample_dir, f))

#shutil.copy(os.path.join(base_dir, '45473537ad2c61d0c9cbeedeb80f93fb9bce8b89.json'), os.path.join(sample_dir,'45473537ad2c61d0c9cbeedeb80f93fb9bce8b89.json'))
