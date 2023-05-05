from pyspark.sql import SparkSession
from transformers import BloomTokenizerFast
import json
from preprocess_func import *
from tqdm import tqdm

spark = SparkSession.builder\
    .appName("Pretrained Data Pipeline")\
    .master("local[*]")\
    .getOrCreate()
sc = spark.sparkContext

log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
LOGGER.info("pyspark script logger initialized")
sc.setLogLevel("ERROR")

max_ent_num = 30
max_rel_num = 20
max_length = 2048

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
src_files = "/harddisk/user/keminglu/pretrained_data_wikipedia_with_mention_eval/*"
output_dir = "/harddisk/user/keminglu/pretrained_data_processed/wikipedia_with_mention_wo_title_simplified_eval/"

def token_length(text, tokenizer):
    return len(tokenizer(text)['input_ids'])

LOGGER.error("[Process Log] Processing:")

raw_data = sc.textFile(src_files)

data = raw_data.map(json.loads)\
    .filter(lambda x: 0 < x['n_ents'] <= max_ent_num and x['n_rel'] <= max_rel_num and not x['title'].startswith("List of"))\
    .map(lambda sample: transform(sample, key_map))\
    .filter(lambda sample: token_length(sample["inputs"] + " " + sample["targets"], tokenizer) <= max_length)\
    .map(json.dumps)\
    .saveAsTextFile(output_dir)
