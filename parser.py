import random

from tqdm import tqdm
import json
import os
import shutil
from os import path

#%% Configuration

base = '/root/autodl-tmp/document_parses/pdf_json/'

destination = '/root/autodl-tmp/document_parses/pdf_parsed/'

sample_dir = './cache'

random_start = True   # Start from a random paper, DEBUGGING ONLY!

cutoff = 100   # How many papers to run on, -1 for all, disable random start for large numbers

merge = True    # Merge the file into one or not

copy_merge_to_wd = True

copy_samples = False  # Leave as False

#%% Code

start_from = random.randint(1,10000)

if copy_merge_to_wd and (cutoff>1000):
    prompt = input('File may be too large, are you sure to copy? [y/N]')
    if not prompt in ('y', 'Y'):
        copy_merge_to_wd = False

if os.path.exists(destination):
    shutil.rmtree(destination)
os.makedirs(destination, exist_ok=True)

if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)
os.makedirs(sample_dir, exist_ok=True)


if merge:
    fm = open(os.path.join(destination, 'merged.txt'), 'w')

for filename in tqdm(os.listdir(base)[start_from:start_from+cutoff]):
    with open(os.path.join(base, filename), 'r') as f:
        json_data = json.load(f)
        paper_id = json_data['paper_id']
        title = json_data['metadata']['title']
        abstract = json_data['abstract'][0]['text'] if json_data['abstract'] else ''
        body_text_raw = json_data['body_text']
        body_text_purged = ''
        for segment in body_text_raw:
            raw_text = segment['text']
            for cite in segment['cite_spans']:
                if cite['ref_id']:
                    raw_text = raw_text.replace(cite['text'], '')
            body_text_purged+=raw_text+'\n'
        # print(cite['text'],end=' ') if not random.randint(0,10) else None

    if not merge:
        with open(os.path.join(destination, paper_id+'.txt'), 'w') as f:
            f.write(title)
            f.write('\n')
            f.write(abstract)
            f.write('\n')
            f.write(body_text_purged)
            f.write('\n')

    else:
        fm.write(title)
        fm.write('\n')
        fm.write(abstract)
        fm.write('\n')
        fm.write(body_text_purged)
        fm.write('\n')

if merge:
    fm.close()


if copy_merge_to_wd:
    shutil.copy(os.path.join(destination, 'merged.txt'), os.path.join(sample_dir, 'merged.txt'))

if copy_samples:
    parsed_files = os.listdir(destination)
    random.shuffle(parsed_files)
    for f in parsed_files[:10]:
        shutil.copy(os.path.join(destination, f), os.path.join(sample_dir, f))

# shutil.copy(os.path.join(destination, 'f3cd160d6f257ff433386449d7b1d52f337a193f.txt'), os.path.join('file_samples','f3cd160d6f257ff433386449d7b1d52f337a193f.txt'))
