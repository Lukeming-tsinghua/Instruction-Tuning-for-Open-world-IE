import os
import json
from multiprocessing.pool import Pool
from preprocess_func import *
from tqdm import tqdm


print("Loading input files...")
data_dir = "/harddisk/data/nlp_data/kb/wikipedia/subset/"
files = os.listdir(data_dir)


output_dir = "/harddisk/user/keminglu/pretrained_data_wikipedia_with_mention_eval"
if not os.path.exists(output_dir):
	print(f"creating output dir:{output_dir}")
	os.makedirs(output_dir)


print("Initializing preprocessor...")
property_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/property_names.json"
entity_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20230301/mapping/sitelinks.enwiki.title.json"
ent_type_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20230301/mapping/p31.json"
mongodb_config = {"host": '9.109.142.31', "port": 27017}
preprocess = Preprocess(
        mongodb_config,
        entity_mapping_file_path,
        property_mapping_file_path,
        ent_type_mapping_file_path,
        dbname='wikidata-20230301'
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