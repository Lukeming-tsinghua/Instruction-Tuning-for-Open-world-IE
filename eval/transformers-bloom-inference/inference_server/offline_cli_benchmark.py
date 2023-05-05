import argparse
import json
import sys
import os
from tqdm import tqdm

from .model_handler import ModelDeployment
from .utils import get_argument_parser, parse_args, print_rank_0, collect_precalculate_entities


BENCHMARK_CONFIG = {
    "entity_typing": {
        "ufet": "ufet_test"
    },
    "entity_linking": {
        "ace2004": "ace2004-test-kilt.jsonl",
        "aida": "aida-test-kilt.jsonl",
        "aquaint": "aquaint-test-kilt.jsonl",
        "clueweb": "clueweb-test-kilt.jsonl",
        "msnbc": "msnbc-test-kilt.jsonl",
    },
    "ner": {
        "conllpp": "conllpp.jsonl",
        "conllpp_gold": "conllpp_gold.jsonl",
        "conllpp_large": "conllpp_large.jsonl",
        "crossner_ai": "crossner_ai_none.jsonl",
        "crossner_ai_gold": "crossner_ai_gold.jsonl",
        "crossner_ai_large": "crossner_ai_large.jsonl",
        "crossner_literature": "crossner_literature_none.jsonl",
        "crossner_literature_gold": "crossner_literature_gold.jsonl",
        "crossner_literature_large": "crossner_literature_large.jsonl",
        "crossner_music_gold": "crossner_music_gold.jsonl",
        "crossner_politics_gold": "crossner_politics_gold.jsonl",
        "crossner_science_gold": "crossner_science_gold.jsonl",
    },
    "relation_extraction": {
        "retacred": "retacred_test.jsonl",
        "retacred_force": "retacred_test_force.jsonl",
        "redocred": "redocred_test.jsonl",
    }
}

RESOURCE_DIR = "/harddisk/user/keminglu/evaluation_corpus/resources"

TYPE_TRIE_CONFIG = {
    "conllpp": os.path.join(RESOURCE_DIR, "basic_types_trie_dict_bloom.pkl"), 
    "conllpp_gold": os.path.join(RESOURCE_DIR, "basic_types_trie_dict_bloom.pkl"), 
    "conllpp_large": os.path.join(RESOURCE_DIR, "basic_types_trie_dict_bloom.pkl"), 
    "crossner_ai_gold": os.path.join(RESOURCE_DIR, "crossner_ai_types_trie_dict_bloom.pkl"),
    "crossner_literature_gold": os.path.join(RESOURCE_DIR, "crossner_literature_types_trie_dict_bloom.pkl"),
    "crossner_music_gold": os.path.join(RESOURCE_DIR, "crossner_music_types_trie_dict_bloom.pkl"),
    "crossner_politics_gold": os.path.join(RESOURCE_DIR, "crossner_politics_types_trie_dict_bloom.pkl"),
    "crossner_science_gold": os.path.join(RESOURCE_DIR, "crossner_science_types_trie_dict_bloom.pkl"),
}

def get_args() -> argparse.Namespace:
    parser = get_argument_parser()
    args = parse_args(parser)
    return args


def main() -> None:
    args = get_args()

    task = args.task
    dataset = args.dataset
    if dataset in TYPE_TRIE_CONFIG:
        args.type_trie_path = TYPE_TRIE_CONFIG[dataset]

    model = ModelDeployment(args, True)


    generate_kwargs = args.generate_kwargs

    data_dir = f"/harddisk/user/keminglu/evaluation_corpus/processed_benchmarks/{task}"
    data_name = BENCHMARK_CONFIG[task][dataset]
    with open(os.path.join(data_dir, data_name)) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    
    trues, preds =[], []
    decode_error_cnt = 0
    for each in tqdm(data):
        string, instruct = each['input_text'].split("\n\n")

        input_text = ('"' + string + '"\n\n' + instruct + each['prompt']).strip()
        prompt = each['prompt']
        true = each['true']
        response = model.generate(text=[input_text], generate_kwargs=generate_kwargs)

        try:
            if task == 'entity_linking':
                pred_ent = json.loads(prompt + response.text[0])['entities'][0]
                pred = [pred_ent['title']] 
                if 'aliases' in pred_ent:
                    pred += pred_ent['aliases']
            elif task == 'entity_typing':
                pred_ent = json.loads(prompt + response.text[0].replace("}]", "}"))
                pred = pred_ent['type']
            elif task == 'ner':
                pred_ent = json.loads(response.text[0])
                pred = [(each['mention'].lower(), each['type'][0]) for each in pred_ent['entities']]
                pred = list(set(pred))
                true = [(mention.lower(), tag) for mention, tag in true]
            elif task == 'relation_extraction':
                pred_ent = json.loads(prompt + response.text[0])
                pred = pred_ent['triplets']
                pred = [(each['head'], each['tail'], list(set([r['title'] for r in each['relations']]))) for each in pred]
            else:
                raise NotImplementedError(f"Benchmark task - {task} - is not implemented.")
            preds.append(pred)
            trues.append(true)
        except (json.JSONDecodeError, KeyError):
            print(input_text)
            print(response.text[0])
            decode_error_cnt += 1
    
    if args.save_results:
        with open(f"results/results_{task}_{dataset}.json", "w") as f:
            json.dump({"task": task, "dataset": dataset, "preds": preds, "trues": trues}, f)

    
    # Evaluation metrics
    if task == "entity_linking":
        pos_true = 0
        for pred, true in zip(preds, trues):
            if len(set(true).difference(set(pred))) == 0:
                pos_true += 1
            else:
                print(pred, true)
        recall = pos_true / len(trues)
        precision = pos_true / len(preds)
        f1 = {2*recall*precision/(precision+recall)}
        print(f"Task: {task}/{dataset}, precision: {precision}, recall: {recall}, Micro F1: {f1}, Decode error rate: {decode_error_cnt/len(data)}")
    elif task == "ner":
        pos_true, n_true, n_pred = 0, 0, 0
        for pred, true in zip(preds, trues):
            pos_true += len(set(true).intersection(set(pred)))
            n_true += len(true)
            n_pred += len(pred)
        recall = pos_true / n_true
        precision = pos_true / n_pred
        f1 = {2*recall*precision/(precision+recall)}
        print(f"Task: {task}/{dataset}, precision: {precision}, recall: {recall}, Micro F1: {f1}, Decode error rate: {decode_error_cnt/len(data)}")

if __name__ == "__main__":
    main()
