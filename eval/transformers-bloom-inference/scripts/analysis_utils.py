import json
import os
from collections import defaultdict
import numpy as np
from rouge import Rouge


def load_data(data_file):
    data = [json.loads(line) for line in open(data_file).readlines()]
    data_split = defaultdict(list)
    for sample in data:
        aug_type = sample['aug_type']
        data_split[aug_type].append(sample)
    return data_split


def load_outputs(sample):
    outputs = sample['outputs'] if 'outputs' in sample else sample['output']
    if outputs and outputs.endswith('.'):
        outputs = outputs.replace(".", "")
    outputs = json.loads(outputs)
    return outputs


def get_decoder_failure_rate(data_split, verbose=False):
    fail_decoder_samples = defaultdict(list)
    for key in data_split:
        fail_decoder_samples[key] = 0
        for i, sample in enumerate(data_split[key]):
            try:
                outputs = load_outputs(sample)
            except (json.JSONDecodeError, TypeError) as e:
                if verbose:
                    print(key, i)
                fail_decoder_samples[key] += 1
    for key in fail_decoder_samples:
        fail_decoder_samples[key] /= len(data_split[key])
    return fail_decoder_samples


def collect_metric(
        all_results,
        task,
        aug_type=None,
        metric=None
    ):
    results = {}
    for key in all_results:
        if task.endswith("_report"):
            if aug_type:
                results[key] = all_results[key][task][aug_type][metric]
            else:
                aug_types = all_results[key][task].keys()
                results[key] = np.mean([all_results[key][task][aug_t][metric] for aug_t in aug_types])

        else:
            if aug_type:
                results[key] = all_results[key][task][aug_type]
            else:
                aug_types = all_results[key][task].keys()
                results[key] = np.mean([all_results[key][task][aug_t] for aug_t in aug_types])
    return results


def get_final_results(all_results, task, aug_type=None, metric=None, neg=False):
    results = collect_metric(all_results,
        task=task,
        aug_type=aug_type,
        metric=metric)
    mean = round(np.mean(list(results.values())), 3) * 100
    if neg:
        mean = 100 - mean
    std = round(np.std(list(results.values())), 3) * 100
    return f"${mean:.1f}_{{{std:.1f}}}$"


### Constraints

def get_ent_num_constraint_report(data_split):
    ent_num_records = defaultdict(lambda: 0)
    for key in data_split:
        if "ent_num" in key or "importance" in key:
            for sample in data_split[key]:

                if type(sample['targets']) == str:
                    sample['targets'] = json.loads(sample['targets'])
                target_entities = sample['targets']['entities']

                try:
                    outputs = load_outputs(sample)
                    output_entities = outputs['entities']
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

                target_len, output_len = len(target_entities), len(output_entities)
                if target_len == output_len:
                    ent_num_records[key] += 1

    for key in ent_num_records:
        ent_num_records[key] /= len(data_split[key])
    return dict(ent_num_records)


def get_ent_type_constraint_report(data_split):
    type_correctness_dict = {}
    for key in data_split:
        if "base_type" in key or "rollup_type" in key:
            type_correctness = 0
            for sample in data_split[key]:

                if key == 'aug_base_type' or key == 'aug_rollup_type':
                    prompt_types = set(sample['aug_info'].split(", "))
                elif key == 'aug_ent_num_and_base_type' or key == 'aug_ent_num_and_rollup_type':
                    prompt_types = set(sample['aug_info'][1].split(", "))
                else:
                    raise RuntimeError("Error")

                try:
                    outputs = load_outputs(sample)
                    output_entities = outputs['entities']
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

                all_flag = []
                for ent in output_entities:
                    flag = False
                    if 'type' in ent:
                        try:
                            for t in ent['type']:
                                if t in prompt_types:
                                    flag = True
                        except TypeError:
                            pass
                    else:
                        flag = True
                    all_flag.append(flag)
                if all(all_flag):
                    type_correctness += 1
                else:
                    #print(prompt_types, output_entities)
                    pass
            type_correctness_dict[key] = type_correctness / len(data_split[key])
    return type_correctness_dict



## Entity Evaluation


def get_partial_ent_evaluation(
        data_split,
        default_mapping,
        verbose=False,
        threshold=0.9
    ):
    R = Rouge()
    results = {}
    for data_key in data_split:

        if 'ent_num' not in data_key:
            continue

        report = {
            "mention_pos_true": 0,
            "title_pos_true": 0,
            "pred_num": 0,
        }

        for orig_sample in data_split[data_key]:
            try:
                sample = default_mapping[orig_sample["id"]]
            except KeyError:
                continue
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            all_mentions = {each['mention']: each['title'] for each in targets["entities"]}

            try:
                outputs = load_outputs(sample)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            
            if 'entities' not in outputs:
                continue

            for each in outputs["entities"]:
                if 'mention' not in each or 'title' not in each:
                    continue
                    
                report['pred_num'] += 1

                mention = each['mention']
                if type(mention) == list:
                    mention = mention[0]
                if mention in all_mentions:
                    report["mention_pos_true"] += 1

                    pred_title = each['title']
                    true_title = all_mentions[mention]

                    if pred_title and type(pred_title) == str:
                        rouge_score = R.get_scores(pred_title, true_title)[0]['rouge-l']['f']
                        if rouge_score >= threshold:
                            report["title_pos_true"] += 1

        report['mention_precision'] = report['mention_pos_true'] / report['pred_num']
        report['title_precision'] = report['title_pos_true'] / report['pred_num']

        results[data_key] = report
    return results

def get_ent_evaluation(
        data_split,
        verbose=False,
        threshold=0.9
    ):
    R = Rouge()
    results = {}
    for data_key in data_split:

        if 'ent_num' in data_key:
            continue

        report = {
            "mention_pos_true": 0,
            "title_pos_true": 0,
            "title_w_aliases_pos_true": 0,
            "true_num": 0,
            "pred_num": 0,
        }

        for sample in data_split[data_key]:
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            all_mentions = {each['mention']: each['title'] for each in targets["entities"]}
            all_aliases = {each['mention']: each['aliases'] if 'aliases' in each else [] for each in targets["entities"]}

            report['true_num'] += len(all_mentions)

            try:
                outputs = load_outputs(sample)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            
            if 'entities' not in outputs:
                continue

            for each in outputs["entities"]:
                if 'mention' not in each or 'title' not in each:
                    continue
                    
                report['pred_num'] += 1

                mention = each['mention']
                if type(mention) == list:
                    mention = mention[0]
                if mention in all_mentions:
                    report["mention_pos_true"] += 1

                    pred_title = each['title']
                    true_title = all_mentions[mention]

                    if pred_title and type(pred_title) == str:
                        rouge_score = R.get_scores(pred_title, true_title)[0]['rouge-l']['f']
                        if rouge_score >= threshold:
                            report["title_pos_true"] += 1
                            report["title_w_aliases_pos_true"] += 1
                        else:
                            aliases = all_aliases[mention]
                            aliases_rouge_scores = [R.get_scores(pred_title, alias)[0]['rouge-l']['f'] for alias in aliases]
                            if aliases_rouge_scores and max(aliases_rouge_scores) >= threshold:
                                report["title_w_aliases_pos_true"] += 1

        report['mention_precision'] = report['mention_pos_true'] / report['pred_num']
        report['mention_recall'] = report['mention_pos_true'] / report['true_num']
        report['mention_f1'] = 2 * report['mention_precision'] * report['mention_recall'] / (report['mention_precision'] + report['mention_recall'])

        report['title_precision'] = report['title_pos_true'] / report['pred_num']
        report['title_recall'] = report['title_pos_true'] / report['true_num']
        report['title_f1'] = 2 * report['title_precision'] * report['title_recall'] / (report['title_precision'] + report['title_recall'])

        report['title_w_aliases_precision'] = report['title_w_aliases_pos_true'] / report['pred_num']
        report['title_w_aliases_recall'] = report['title_w_aliases_pos_true'] / report['true_num']
        report['title_w_aliases_f1'] = 2 * report['title_w_aliases_precision'] * report['title_w_aliases_recall'] / (report['title_w_aliases_precision'] + report['title_w_aliases_recall'])

        results[data_key] = report
    return results


def get_ent_generalization_evaluation(
        data_split,
        verbose=False,
        threshold=0.9
    ):
    R = Rouge()
    results = {}
    for data_key in data_split:

        if 'ent_num' in data_key:
            continue

        report = {
            "unseen_mention_recall": 0,
            "unseen_title_recall": 0,
            "unseen_title_recall_w_aliases": 0,
            "unseen_mention_num": 0,
            "seen_mention_recall": 0,
            "seen_title_recall": 0,
            "seen_title_recall_w_aliases": 0,
            "seen_mention_num": 0,
        }

        for sample in data_split[data_key]:
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            unseen_mentions = {each['mention']: each['title'] for each in targets["entities"] if each['ood'] != 'in'}
            unseen_aliases = {each['mention']: each['aliases'] if 'aliases' in each else [] for each in targets["entities"] if each['ood'] != 'in'}
            seen_mentions = {each['mention']: each['title'] for each in targets["entities"] if each['ood'] == 'in'}
            seen_aliases = {each['mention']: each['aliases'] if 'aliases' in each else [] for each in targets["entities"] if each['ood'] == 'in'}

            report['unseen_mention_num'] += len(unseen_mentions)
            report['seen_mention_num'] += len(seen_mentions)

            try:
                outputs = load_outputs(sample)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            
            if 'entities' not in outputs:
                continue

            for each in outputs["entities"]:
                if 'mention' not in each or 'title' not in each:
                    continue

                mention = each['mention']
                if type(mention) == list:
                    mention = mention[0]
                if mention in unseen_mentions:
                    status = 'unseen'
                elif mention in seen_mentions:
                    status = 'seen'
                else:
                    continue

                report[f"{status}_mention_recall"] += 1
                pred_title = each['title']
                true_title = unseen_mentions[mention] if status == 'unseen' else seen_mentions[mention]

                if pred_title and type(pred_title) == str:
                    rouge_score = R.get_scores(pred_title, true_title)[0]['rouge-l']['f']
                    if rouge_score >= threshold:
                        report[f"{status}_title_recall"] += 1
                        report[f"{status}_title_recall_w_aliases"] += 1
                    else:
                        aliases = unseen_aliases[mention] if status == 'unseen' else seen_aliases[mention]
                        aliases_rouge_scores = [R.get_scores(pred_title, alias)[0]['rouge-l']['f'] for alias in aliases]
                        if aliases_rouge_scores and max(aliases_rouge_scores) >= threshold:
                            report[f"{status}_title_recall_w_aliases"] += 1

        for status in ('unseen', 'seen'):
            for key in [f"{status}_mention_recall", f"{status}_title_recall", f"{status}_title_recall_w_aliases"]:
                report[key] = report[key] / report[f"{status}_mention_num"]

        results[data_key] = report
    return results


def get_entity_info_correctness(data_split):
    R = Rouge()
    report = {}
    for key in data_split:
        results = {
            "desc_rouge": [],
            "unseen_desc_rouge": [],
            "seen_desc_rouge": [],
            "aliases_pos_true": 0,
            "aliases_pred_num": 0,
            "aliases_true_num": 0,
            "unseen_aliases_pos_true": 0,
            "unseen_aliases_pred_num": 0,
            "unseen_aliases_true_num": 0,
            "seen_aliases_pos_true": 0,
            "seen_aliases_pred_num": 0,
            "seen_aliases_true_num": 0,
        }

        for sample in data_split[key]:
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            target_dict = {
                (each['mention'], each['title']): (
                    each['description'] if 'description' in each else '',
                    each['aliases'] if 'aliases' in each else [],
                    each['ood']
                ) for each in targets['entities']}

            try:
                outputs = load_outputs(sample)
                output_entities = outputs['entities']
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
                
            for ent in output_entities:

                if 'mention' in ent and 'title' in ent:
                    mention, title = ent['mention'], ent['title']
                    # fix error in ChatGPT generations
                    if type(mention) == list:
                        mention = mention[0]
                    if type(title) != str:
                        continue

                    if (mention, title) in target_dict:
                        target_description, target_aliases, ood_flag = target_dict[(mention, title)]
                        pred_description = ent['description'] if 'description' in ent else ''

                        try:
                            pred_aliases = set(ent['aliases'] if 'aliases' in ent and ent['aliases'] else [])
                        except Exception as e:
                            pred_aliases = set([])

                        target_aliases = set(target_aliases)

                        if ood_flag == 'in':
                            status = "seen"
                        else:
                            status = "unseen"
                        
                        if pred_description and target_description and pred_description != '' and target_description != '' and type(pred_description) == type(target_description):
                            results["desc_rouge"].append(R.get_scores(pred_description, target_description)[0]['rouge-l']['f'])
                            results[f"{status}_desc_rouge"].append(R.get_scores(pred_description, target_description)[0]['rouge-l']['f'])

                        results["aliases_pos_true"] += len(pred_aliases.intersection(target_aliases))
                        results["aliases_pred_num"] += len(pred_aliases)
                        results["aliases_true_num"] += len(target_aliases)
                        results[f"{status}_aliases_pos_true"] += len(pred_aliases.intersection(target_aliases))
                        results[f"{status}_aliases_pred_num"] += len(pred_aliases)
                        results[f"{status}_aliases_true_num"] += len(target_aliases)

        results["desc_rouge"] = np.mean(results["desc_rouge"])
        results["aliases_p"] = results["aliases_pos_true"] / results["aliases_pred_num"] if results["aliases_pred_num"] != 0 else 0
        results["aliases_r"] = results["aliases_pos_true"] / results["aliases_true_num"] if results["aliases_true_num"] != 0 else 0
        results["aliases_f"] = 2 * results["aliases_p"] * results["aliases_r"] / (results["aliases_p"] + results["aliases_r"]) if results["aliases_p"] + results["aliases_r"] != 0 else 0
        for status in ("seen", "unseen"):
            results[f"{status}_aliases_p"] = results[f"{status}_aliases_pos_true"] / results[f"{status}_aliases_pred_num"] if results[f"{status}_aliases_pred_num"] != 0 else 0
            results[f"{status}_aliases_r"] = results[f"{status}_aliases_pos_true"] / results[f"{status}_aliases_true_num"] if results[f"{status}_aliases_true_num"] != 0 else 0
            if results[f"{status}_aliases_p"] + results[f"{status}_aliases_r"] != 0:
                results[f"{status}_aliases_f"] = 2 * results[f"{status}_aliases_p"] * results[f"{status}_aliases_r"] / (results[f"{status}_aliases_p"] + results[f"{status}_aliases_r"])
            results[f"{status}_desc_rouge"] = np.mean(results[f"{status}_desc_rouge"])

        report[key] = results
    return report


## Entity Typing

def get_ent_type_evaluation(data_split):
    report = {}
    for key in data_split:
        pos_true = 0
        pred_num = 0
        true_num = 0

        for sample in data_split[key]:

            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']
            target_type_dict = {(each['mention'], each['title']): each['type'] for each in targets['entities'] if 'type' in each}

            for _, value in target_type_dict.items():
                true_num += len(value)

            try:
                outputs = load_outputs(sample)
                output_entities = outputs['entities']
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            for ent in output_entities:
                try:
                    if 'mention' in ent and 'title' in ent:
                        pred_types = set(ent['type'] if 'type' in ent else [])
                        pred_num += len(pred_types) 
                        if (ent['mention'], ent['title']) in target_type_dict:
                            target_types = target_type_dict[(ent['mention'], ent['title'])]
                            target_types = set(target_types)
                            pos_true += len(pred_types.intersection(target_types))
                except TypeError:
                    continue

        p = pos_true / pred_num
        r = pos_true / true_num
        f = 2 * p * r / (p + r) if p + r != 0 else 0
        report[key] = {"p": p, "r": r, "f": f}
    return report


def get_ent_type_generalization_evaluation(data_split):
    report = {}
    for key in data_split:
        results = {
            "seen_pos_true": 0,
            "seen_pred_num": 0,
            "seen_true_num": 0,
            "unseen_pos_true": 0,
            "unseen_pred_num": 0,
            "unseen_true_num": 0,
        }
        for sample in data_split[key]:

            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']
            target_type_dict = {(each['mention'], each['title']): (each['type'], each['ood']) for each in targets['entities'] if 'type' in each}

            for _, value in target_type_dict.items():
                status = "seen" if value[1] == "in" else "unseen"
                results[f"{status}_true_num"] += len(value[0])

            try:
                outputs = load_outputs(sample)
                output_entities = outputs['entities']
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            

            for ent in output_entities:
                try:
                    if 'mention' in ent and 'title' in ent:
                        if (ent['mention'], ent['title']) in target_type_dict:
                            pred_types = set(ent['type'] if 'type' in ent else [])
                            target_types, ood_flag = target_type_dict[(ent['mention'], ent['title'])]
                            target_types = set(target_types)

                            if ood_flag == 'in':
                                status = "seen"
                            else:
                                status = "unseen"

                            results[f"{status}_pos_true"] += len(pred_types.intersection(target_types))
                            results[f"{status}_pred_num"] += len(pred_types) 
                except TypeError:
                    continue

        for status in ('unseen', 'seen'):
            results[f"{status}_p"] = results[f"{status}_pos_true"] / results[f"{status}_pred_num"]
            results[f"{status}_r"] = results[f"{status}_pos_true"] / results[f"{status}_true_num"]
            if results[f"{status}_p"] + results[f"{status}_r"] != 0:
                results[f"{status}_f"] = 2 * results[f"{status}_p"] * results[f"{status}_r"] / (results[f"{status}_p"] + results[f"{status}_r"])
            else:
                results[f"{status}_f"] = 0
        report[key] = results
    return report

## Utils

def get_default_mapping_dict(data_split):
    default_mapping_dict = {}
    for sample in data_split['aug_default']:
        default_mapping_dict[sample['id']] = sample
    return default_mapping_dict


## Open Relation Extraction

def get_open_relation_extraction_report(data_split, data_file, carb_dir="./CaRB/files", collect_new_relations=False):
    report = {}
    new_relations = []
    for key in data_split:
        data_for_carb = []
        if 'ent_num' in key:
            continue
        pos_true, pred_num, true_num = 0, 0, 0
        for sample in data_split[key]:
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            try:
                outputs = load_outputs(sample)
                output_entities = outputs['entities']
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            true_mention_dict = {each['mention']: (each['title'], each['ood']) for each in targets['entities']}
            try:
                pred_mention_dict = {each['mention']: {
                    'title': each['title'] if 'title' in each else '' , 
                    'type': each['type'] if 'type' in each else [] , 
                    'description': each['description'] if 'description' in each else '',
                    'aliases': each['aliases'] if 'aliases' in each else [] , 
                    } for each in output_entities if 'mention' in each and 'title' in each}
            except (KeyError, TypeError):
                continue

            true_triplets = targets['triplets']
            pred_triplets = outputs['triplets'] if 'triplets' in outputs else []

            true_relations, pred_relations = [], []
            for triplet in true_triplets:
                for rel in triplet['relations']:
                    true_relations.append((triplet['head'], triplet['tail'], rel))
            for triplet in pred_triplets:
                if 'relations' in triplet:
                    for rel in triplet['relations']:
                        if 'head' in triplet and 'tail' in triplet and rel != '' and rel:
                            if type(triplet['head']) == type(triplet['tail']) == type(rel) == str and triplet['head'] != '' and triplet['tail'] != '' and triplet['head'] and triplet['tail']:
                                pred_relations.append((triplet['head'], triplet['tail'], rel if "_" not in rel else rel.replace("_", " ")))

            true_relations = set(true_relations)
            pred_relations = set(pred_relations)

            if len(true_relations) > 0:
                data_for_carb.append((sample['inputs'], list(true_relations), list(pred_relations)))

            # Collect New Rels
            if collect_new_relations:
                new_rels = pred_relations.difference(true_relations)
                for head, tail, rel in new_rels:
                    new_relations.append({
                        "context": sample["inputs"],
                        "head_m": head,
                        "tail_m": tail,
                        "pred_head_title": pred_mention_dict[head]['title'] if head in pred_mention_dict else False,
                        "pred_tail_title": pred_mention_dict[tail]['title'] if tail in pred_mention_dict else False,
                        "pred_head_info": pred_mention_dict[head] if head in pred_mention_dict else False,
                        "pred_tail_info": pred_mention_dict[tail] if tail in pred_mention_dict else False,
                        "true_head_title": true_mention_dict[head] if head in true_mention_dict else None,
                        "true_tail_title": true_mention_dict[tail] if tail in true_mention_dict else None,
                        "rel": rel
                    })

            pos_true += len(true_relations.intersection(pred_relations))
            pred_num += len(pred_relations)
            true_num += len(true_relations)

        # CaRB score
        gold_path = os.path.join(carb_dir, f"{data_file}_gold_{key}.tsv")
        pred_path = os.path.join(carb_dir, f"{data_file}_pred_{key}.tsv")
        out_path = os.path.join(carb_dir, f"{data_file}_out_{key}.tsv")
        with open(gold_path, "w") as fg:
            with open(pred_path, "w") as fp:
                for sent, gold, pred in data_for_carb:
                    sent = sent.replace("\t", "")
                    for h, t, r in gold:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", "")
                        line = f"{sent}\t{r}\t{h}\t{t}\n"
                        fg.write(line)
                    for h, t, r in pred:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", " ")
                        line = f"{sent}\t1\t{r}\t{h}\t{t}\n"
                        fp.write(line)

        process = os.popen(f"python ./CaRB/carb.py --gold=\"{gold_path}\" --out=\"{out_path}\" --tabbed=\"{pred_path}\"")
        results = process.read()
        process.close()
        try:
            carb_p, carb_r, carb_f1 = [float(each) for each in results.split(": ")[-1].strip()[1:-2].split(" ") if each != '']
        except ValueError as e:
            print(results)

        # EM Score
        p = pos_true / pred_num if pred_num != 0 else 0
        r = pos_true / true_num if true_num != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        report[key] = {"em_p": p, "em_r": r, "em_f": f1, "carb_p": carb_p, "carb_r": carb_r, "carb_f1": carb_f1}

    return report, new_relations


def get_open_relation_extraction_generalization_report(data_split, data_file, carb_dir="./CaRB/files"):
    report = {}

    for key in data_split:
        unseen_data_for_carb, seen_data_for_carb = [], []
        if 'ent_num' in key:
            continue
        unseen_pos_true, seen_pos_true, pred_num, unseen_true_num, seen_true_num = 0, 0, 0, 0, 0
        for sample in data_split[key]:
            targets = json.loads(sample['targets']) if type(sample['targets']) == str else sample['targets']

            try:
                outputs = load_outputs(sample)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            true_mention_dict = {each['mention']: (each['title'], each['ood']) for each in targets['entities']}
            unseen_true_triplets, seen_true_triplets = [], []
            for each in targets['triplets']:
                if true_mention_dict[each['head']][1] != 'in' or \
                    true_mention_dict[each['tail']][1] != 'in':
                    unseen_true_triplets.append(each)
                else:
                    seen_true_triplets.append(each)

            pred_triplets = outputs['triplets'] if 'triplets' in outputs else []

            unseen_true_relations, seen_true_relations, pred_relations = [], [], []
            for triplet in unseen_true_triplets:
                for rel in triplet['relations']:
                    item = (triplet['head'], triplet['tail'], rel)
                    #if item not in train_triplets:
                    unseen_true_relations.append(item)
            for triplet in seen_true_triplets:
                for rel in triplet['relations']:
                    item = (triplet['head'], triplet['tail'], rel)
                    #if item not in train_triplets:
                    seen_true_relations.append(item)
            
            for triplet in pred_triplets:
                if 'relations' in triplet:
                    for rel in triplet['relations']:
                        if 'head' in triplet and 'tail' in triplet:
                            if type(triplet['head']) == type(triplet['tail']) == type(rel) == str and triplet['head'] != '' and triplet['tail'] != '':
                                pred_relations.append((triplet['head'], triplet['tail'], rel if "_" not in rel else rel.replace("_", " ")))

            unseen_true_relations = set(unseen_true_relations)
            seen_true_relations = set(seen_true_relations)
            pred_relations = set(pred_relations)

            if len(unseen_true_relations) > 0:
                unseen_data_for_carb.append((sample['inputs'], list(unseen_true_relations), list(pred_relations)))
            if len(seen_true_relations) > 0:
                seen_data_for_carb.append((sample['inputs'], list(seen_true_relations), list(pred_relations)))

            unseen_pos_true += len(unseen_true_relations.intersection(pred_relations))
            unseen_true_num += len(unseen_true_relations)
            seen_pos_true += len(seen_true_relations.intersection(pred_relations))
            seen_true_num += len(seen_true_relations)
            pred_num += len(pred_relations)

        # CaRB score
        unseen_gold_path = os.path.join(carb_dir, f"{data_file}_unseen_gold_{key}.tsv")
        seen_gold_path = os.path.join(carb_dir, f"{data_file}_seen_gold_{key}.tsv")
        unseen_out_path = os.path.join(carb_dir, f"{data_file}_unseen_out_{key}.tsv")
        seen_out_path = os.path.join(carb_dir, f"{data_file}_seen_out_{key}.tsv")
        pred_path = os.path.join(carb_dir, f"{data_file}_pred_{key}.tsv")

        with open(unseen_gold_path, "w") as fg:
            with open(pred_path, "w") as fp:
                for sent, gold, pred in unseen_data_for_carb:
                    sent = sent.replace("\t", "")
                    for h, t, r in gold:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", "")
                        fg.write(f"{sent}\t{r}\t{h}\t{t}\n")
                    for h, t, r in pred:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", " ")
                        fp.write(f"{sent}\t1\t{r}\t{h}\t{t}\n")
        with open(seen_gold_path, "w") as fg:
            with open(pred_path, "w") as fp:
                for sent, gold, pred in seen_data_for_carb:
                    sent = sent.replace("\t", "")
                    for h, t, r in gold:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", "")
                        fg.write(f"{sent}\t{r}\t{h}\t{t}\n")
                    for h, t, r in pred:
                        h, t, r = h.replace("\t", "").replace("\n", ""), t.replace("\t", "").replace("\n", ""), r.replace("\t", "").replace("\n", "")
                        r = r.replace("_", " ")
                        fp.write(f"{sent}\t1\t{r}\t{h}\t{t}\n")

        process = os.popen(f"python ./CaRB/carb.py --gold=\"{unseen_gold_path}\" --out=\"{unseen_out_path}\" --tabbed=\"{pred_path}\"")
        results = process.read()
        process.close()
        try:
            unseen_carb_r = [float(each) for each in results.split(": ")[-1].strip()[1:-2].split(" ") if each != ''][1]
        except ValueError as e:
            print(results)

        process = os.popen(f"python ./CaRB/carb.py --gold=\"{seen_gold_path}\" --out=\"{seen_out_path}\" --tabbed=\"{pred_path}\"")
        results = process.read()
        process.close()
        try:
            seen_carb_r = [float(each) for each in results.split(": ")[-1].strip()[1:-2].split(" ") if each != ''][1]
        except ValueError as e:
            print(results)

        # EM Score
        unseen_r = unseen_pos_true / unseen_true_num if unseen_true_num != 0 else 0
        seen_r = seen_pos_true / seen_true_num if seen_true_num != 0 else 0
        report[key] = {"unseen_em_r": unseen_r, "seen_em_r": seen_r, "unseen_carb_r": unseen_carb_r, "seen_carb_r": seen_carb_r, "unseen_true_num": unseen_true_num, "seen_true_num": seen_true_num}

    return report