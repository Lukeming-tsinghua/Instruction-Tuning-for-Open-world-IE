from pyspark.sql import SparkSession
import os
import json
from augment_utils import augment_sample

spark = SparkSession.builder\
    .appName("Pretrained Data Pipeline")\
    .master("local[*]")\
    .getOrCreate()
sc = spark.sparkContext

log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)
LOGGER.info("pyspark script logger initialized")
sc.setLogLevel("ERROR")

LOGGER.error("[Process Log] Loading mapping files:")
mapping_dir = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/"
ontology = json.load(open(os.path.join(mapping_dir, "qid2p279.json")))
qid2label = json.load(open(os.path.join(mapping_dir, "qid2sitelinks.enwiki.title.json")))
label2qid = {value: key for key, value in qid2label.items()}
prior_map = json.load(open("/harddisk/data/nlp_data/kb/wikipedia/20220620/enwiki-20220620/output/mention/entity_prior.json"))

ontology = sc.broadcast(ontology)
qid2label = sc.broadcast(qid2label)
label2qid = sc.broadcast(label2qid)
prior_map = sc.broadcast(prior_map)

src_file = "/harddisk/user/keminglu/pretrained_data_processed/wikipedia_with_mention_wo_title_simplified/corpus"
output_dir = "/harddisk/user/keminglu/pretrained_data_processed/wikipedia_with_mention_wo_title_simplified_aug"
raw_data = sc.textFile(src_file)

LOGGER.error("[Process Log] Processing:") 
raw_data.map(json.loads)\
        .flatMap(lambda x: augment_sample(x, prior_map.value, label2qid.value, qid2label.value, ontology.value)).map(json.dumps).saveAsTextFile(output_dir)