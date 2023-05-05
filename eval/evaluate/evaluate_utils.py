import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import json
from collections import defaultdict
from pymongo import MongoClient
from dataclasses import dataclass
from enum import Enum


def infer(
    context,
    prompt,
    model,
    tokenizer,
    device,
    num_beams=4,
    do_sample=False,
    length_penalty=5,
    max_new_tokens=2048,
    skip_special_tokens=True
):

    inputs = tokenizer(
        context + prompt, return_tensors="pt")['input_ids'].to(device)
    max_new_tokens = max_new_tokens - len(inputs[0])

    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs,
            num_beams=num_beams,
            do_sample=do_sample,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens)

    prompt_length = len(
        tokenizer.decode(
            inputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    )
    texts = tokenizer.batch_decode(
        outputs, skip_special_tokens=skip_special_tokens)

    def cut_prompt(text, prompt_length):
        index = text.find('{"entities":')
        if index != -1:
            text = text[index:].strip()
        else:
            text = text[prompt_length:].strip()
        return text
    
    texts = [cut_prompt(text, prompt_length) for text in texts]
    return texts


class KGExtractor(object):
    def __init__(self,
                 mongodb_config,
                 entity_mapping_file_path,
                 property_mapping_file_path,
                 topk_aliases=5,
                 langs={'en': 'english'}
                 ):

        self.entity_mapping = json.load(open(entity_mapping_file_path))
        self.property_mapping = json.load(open(property_mapping_file_path))
        self.entity_mapping_inverse = {
            v: k for k, v in self.entity_mapping.items()}
        self.property_mapping_inverse = {
            v: k for k, v in self.property_mapping.items()}

        self.client = MongoClient(**mongodb_config)
        self.kg_collection = self.client['wikidata']['kg']
        self.raw_collection = self.client['wikidata']['raw']

        self.topk_aliases = topk_aliases
        self.langs = langs

    def entity2id(self, entity_name):
        if entity_name in self.entity_mapping_inverse:
            return self.entity_mapping_inverse[entity_name]
        else:
            return None

    def property2id(self, property_name):
        if property_name in self.property_mapping_inverse:
            return self.property_mapping_inverse[property_name]
        else:
            return None

    def collect_wikidata_context(self, wiki_id):
        query = {'id': wiki_id}
        res = self.raw_collection.find_one(query)
        output = {"description": {}, "aliases": {}}
        for lang in self.langs:
            if 'descriptions' in res and lang in res['descriptions']:
                output["description"][self.langs[lang]
                                      ] = res['descriptions'][lang]['value']
            if 'aliases' in res and lang in res['aliases']:
                output["aliases"][self.langs[lang]] = [each['value']
                                                       for each in res['aliases'][lang]][:self.topk_aliases]
        return output

    def collect_wikidata_relation(self, s_id, t_id):
        query = {'s': s_id, 'o': t_id}
        res = self.kg_collection.find(query)
        relations = []
        for item in res:
            s, p, o = item['s'], item['p'], item['o']
            relations.append(p)
        return relations


class Evaluator(object):
    def __init__(self, extractor):
        self.extractor = extractor

    def check_parsing(self, text):
        record, decode_error = None, False
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            decode_error = True
        return record, decode_error

    def deduplicate_dict(self, inputs):
        return list(map(json.loads, set(map(json.dumps, inputs))))

    def check_entities(self, pred, src):
        entities = self.deduplicate_dict(pred['entities'])
        deduplicate_drop_cnt = len(pred['entities']) - len(entities)

        entity_mention_not_in_context_cnt = 0
        entity_mention_not_in_context = []
        mention_to_title_map = {}
        # Ensure mentions are in context
        for entity in entities:
            entity_mention = entity['mention']
            if src.find(entity_mention) == -1:
                entity_mention_not_in_context_cnt += 1
                entity_mention_not_in_context.append(entity_mention)
            mention_to_title_map[entity_mention] = entity['title']

        # Collect aligned entities and unaligned entities in prediction
        entity_evaluate_info = []
        unaligned_evaluate_info = []
        for entity in entities:
            entity_name = entity['title']
            qid = self.extractor.entity2id(entity_name)
            if qid:
                context = self.extractor.collect_wikidata_context(qid)
                entity_evaluate_info.append((entity, context))
            else:
                unaligned_evaluate_info.append(entity)

        # count mapping attributes
        desc_map_cnt = 0
        aliases_map_cnt = 0
        for entity, context in entity_evaluate_info:
            try:
                if entity["description"]["english"] == context["description"]["english"]:
                    desc_map_cnt += 1

                pred_aliases = set(entity["aliases"]["english"])
                true_aliases = set(context["aliases"]["english"])
                overlap = pred_aliases.intersection(true_aliases)
                if len(overlap) == len(true_aliases):
                    aliases_map_cnt += 1
            except KeyError:
                pass

        report = {
            "entity_cnt": len(pred["entities"]),
            "entity_deduplicated_cnt": len(entities),
            "entity_deduplicate_drop_cnt": deduplicate_drop_cnt,
            "entity_mention_not_in_context_cnt": entity_mention_not_in_context_cnt,
            "entity_mention_not_in_context": entity_mention_not_in_context,
            "entity_InKB_cnt": len(entity_evaluate_info),
            "entity_OOD_cnt": len(unaligned_evaluate_info),
            "entity_desc_map_cnt": desc_map_cnt,
            "entity_aliases_map_cnt": aliases_map_cnt,
            "entity_unaligned": unaligned_evaluate_info
        }
        return report, mention_to_title_map

    def check_triplets(self, pred, mention_to_title_map):
        triplets = self.deduplicate_dict(pred["triplets"])
        deduplicate_drop_cnt = len(pred["triplets"]) - len(triplets)

        # split aligned and unaligned entity pair
        triplet_not_aligned_mention_cnt = 0
        triplet_evaluate_info = []
        unaligned_evaluate_info = []
        for triplet in triplets:
            if triplet["head"] in mention_to_title_map and triplet["tail"] in mention_to_title_map:
                s, o = mention_to_title_map[triplet["head"]
                                            ], mention_to_title_map[triplet["tail"]]
                s = self.extractor.entity2id(s)
                o = self.extractor.entity2id(o)
                if s and o:
                    ps = self.extractor.collect_wikidata_relation(s, o)
                    triplet_evaluate_info.append(
                        (s, o, triplet["relations"], ps))
                else:
                    unaligned_evaluate_info.append(triplet)
            else:
                triplet_not_aligned_mention_cnt += 1
                unaligned_evaluate_info.append(triplet)

        relation_cnt_all = []
        relation_In_KB_cnt_all = []
        relation_OOD_cnt_all = []
        relation_desc_map_cnt_all = []
        relation_aliases_map_cnt_all = []
        relation_In_KB_overlap_cnt_all = []
        relation_In_KB_missing_cnt_all = []
        relation_In_KB_expand_cnt_all = []
        relation_In_KB_missing_all = []
        relation_In_KB_expand_all = []

        for s, o, pred_relations, kb_relations in triplet_evaluate_info:
            # check whther pred_relations are aligned with KB
            relation_evaluate_info = []
            unaligned_relation_info = []
            In_KB_pids = []
            for relation in pred_relations:
                property_name = relation['title']
                pid = self.extractor.property2id(property_name)
                if pid:
                    In_KB_pids.append(pid)
                    property_context = self.extractor.collect_wikidata_context(
                        pid)
                    relation_evaluate_info.append((relation, property_context))
                else:
                    unaligned_relation_info.append(relation)

            relation_cnt_all.append(len(pred_relations))
            relation_In_KB_cnt_all.append(len(In_KB_pids))
            relation_OOD_cnt_all.append(len(unaligned_relation_info))

            # check relation coverage
            relation_In_KB_overlap_cnt_all.append(
                len(set(kb_relations).intersection(In_KB_pids)))
            relation_In_KB_missing_cnt_all.append(
                len(set(kb_relations).difference(In_KB_pids)))
            relation_In_KB_expand_cnt_all.append(
                len(set(In_KB_pids).difference(kb_relations)))

            relation_In_KB_missing_all.append(
                [(s, o, p) for p in set(kb_relations).difference(In_KB_pids)])
            relation_In_KB_expand_all.append(
                [(s, o, p) for p in set(In_KB_pids).difference(kb_relations)])

            # check attribute correctness of In-KB relations
            desc_map_cnt = 0
            aliases_map_cnt = 0
            for relation, property in relation_evaluate_info:
                try:
                    if relation["description"]["english"] == property["description"]["english"]:
                        desc_map_cnt += 1

                    pred_aliases = set(relation["aliases"]["english"])
                    true_aliases = set(property["aliases"]["english"])
                    overlap = pred_aliases.intersection(true_aliases)
                    if len(overlap) == len(true_aliases):
                        aliases_map_cnt += 1
                except KeyError:
                    pass
            relation_desc_map_cnt_all.append(desc_map_cnt)
            relation_aliases_map_cnt_all.append(aliases_map_cnt)

        report = {
            "triplet_cnt": len(pred["triplets"]),
            "triplet_deduplicate_cnt": len(triplets),
            "triplet_deduplicate_drop_cnt": deduplicate_drop_cnt,
            "triplet_not_aligned_mention_cnt": triplet_not_aligned_mention_cnt,
            "triplet_In_KB_cnt": len(triplet_evaluate_info),
            "triplet_OOD_cnt": len(unaligned_evaluate_info),

            "relation_cnt": sum(relation_cnt_all),
            "relation_In_KB_cnt": sum(relation_In_KB_cnt_all),
            "relation_OOD_cnt": sum(relation_OOD_cnt_all),
            "relation_desc_map_cnt": sum(relation_desc_map_cnt_all),
            "relation_aliases_map_cnt": sum(relation_aliases_map_cnt_all),
            "relation_In_KB_overlap_cnt": sum(relation_In_KB_overlap_cnt_all),
            "relation_In_KB_missing_cnt": sum(relation_In_KB_missing_cnt_all),
            "relation_In_KB_expand_cnt": sum(relation_In_KB_expand_cnt_all),

            "relation_cnt_all": relation_cnt_all,
            "relation_In_KB_cnt_all": relation_In_KB_cnt_all,
            "relation_OOD_cnt_all": relation_OOD_cnt_all,
            "relation_desc_map_cnt_all": relation_desc_map_cnt_all,
            "relation_aliases_map_cnt_all": relation_aliases_map_cnt_all,
            "relation_In_KB_overlap_cnt_all": relation_In_KB_overlap_cnt_all,
            "relation_In_KB_missing_cnt_all": relation_In_KB_missing_cnt_all,
            "relation_In_KB_expand_cnt_all": relation_In_KB_expand_cnt_all,
            "relation_In_KB_missing_all": relation_In_KB_missing_all,
            "relation_In_KB_expand_all": relation_In_KB_expand_all
        }

        return report

    def __call__(self, res):
        src, output = res["input"], res["output"]
        pred, decode_error = self.check_parsing(output[0])
        if not decode_error:
            report, mention_to_title_map = self.check_entities(pred, src)
            report.update(self.check_triplets(pred, mention_to_title_map))
            report.update({"status": True})
        else:
            report = {"status": False}
        return report
