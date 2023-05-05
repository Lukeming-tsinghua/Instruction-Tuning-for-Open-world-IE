import os
import json
import logging
from collections import defaultdict
from multiprocessing.pool import Pool
from preprocess_func import *
from tqdm import tqdm


print("Loading input files...")
data_dir = "/harddisk/data/nlp_data/kb/wikipedia/20220620/enwiki-20220620/output/blocks.ann/"
files = os.listdir(data_dir)


output_dir = "/harddisk/user/keminglu/pretrained_data_wikipedia_with_mention"
if not os.path.exists(output_dir):
	print(f"creating output dir:{output_dir}")
	os.makedirs(output_dir)


print("Initializing preprocessor...")
property_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/property_names.json"
entity_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/qid2sitelinks.enwiki.title.json"
ent_type_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/qid2p31.json"
mongodb_config = {"host": '10.12.192.31', "port": 27017}
preprocess = Preprocess(
        mongodb_config,
        entity_mapping_file_path,
        property_mapping_file_path,
        ent_type_mapping_file_path
    )


def init():
    global preprocess


def run(file_path):
    data = []
    with open(os.path.join(data_dir, file_path)) as f:
        line = f.readline()
        while line:
            data.append(preprocess(line))
            line = f.readline()
    return data, file_path


print("Preprocessing files...")
with Pool(32, initializer=init) as pool:
    for output, file_path in tqdm(pool.imap_unordered(run, files)):
        with open(os.path.join(output_dir, file_path), "w") as f:
            for line in output:
                f.write(json.dumps(line) + "\n")
