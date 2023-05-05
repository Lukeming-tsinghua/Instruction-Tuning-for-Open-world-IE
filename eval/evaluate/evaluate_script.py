import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json
from evaluate_utils import *
from tqdm import tqdm



if __name__ == "__main__":

    print("Loading inference configs...")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("/harddisk/user/keminglu/bigscience_tokenizer")
    model = AutoModelForCausalLM.from_pretrained("/data/home/keminglu/workspace/dev-cloud/finetune_1b1_data_v0_epoch_1").to(device)

    property_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/property_names.json"
    entity_mapping_file_path = "/harddisk/data/nlp_data/kb/wikidata/20210520/mapping/qid2sitelinks.enwiki.title.json"
    mongodb_config = {"host": '10.12.192.31', "port": 27017}
    extractor = KGExtractor(mongodb_config, entity_mapping_file_path, property_mapping_file_path)
    evaluator = Evaluator(extractor)

    print("Loading data...")
    corpus_file_path = "/harddisk/user/keminglu/evaluation_corpus/wiki_eval.txt"
    output_file_path = corpus_file_path.replace(".txt", "_output.txt")
    report_file_path = corpus_file_path.replace(".txt", "_output_with_evaluation.txt")
    data = open(corpus_file_path).readlines()

    print("Inferring...")
    all_res = []
    data = data[:1000]
    pbar = tqdm(total=len(data))
    for each in data:
        res = infer(each, "", model, tokenizer, device)
        all_res.append({"input": each, "output": res})
        pbar.update(1)

    with open(output_file_path, "w") as f:
        for res in all_res:
            f.write(json.dumps(res) + "\n")

    with open(output_file_path) as f:
        outputs = [json.loads(line) for line in f.readlines()]
    
    with open(report_file_path, "w") as f:
        for output in outputs:
            report = evaluator(output['output'])
            output.update({"evaluation": report})
            f.write(json.dumps(output) + "\n")