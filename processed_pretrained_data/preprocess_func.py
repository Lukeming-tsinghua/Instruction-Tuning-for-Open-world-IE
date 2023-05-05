import os
import json
from collections import defaultdict, OrderedDict
from pymongo import MongoClient


class Preprocess(object):
    def __init__(self,
                 mongodb_config,
                 entity_mapping_file_path,
                 property_mapping_file_path,
                 ent_type_mapping_file_path,
                 abstract_only=True,
                 eval_extract=False,
                 topk_aliases=5,
                 topk_ent_type=3,
                 langs={'en': 'english'},
                 dbname='wikidata'
                 ):

        self.abstract_only = abstract_only
        self.eval_extract = eval_extract
        if self.abstract_only == self.eval_extract:
            raise ValueError("abstract_only should be different from eval_extract")

        self.entity_mapping = json.load(open(entity_mapping_file_path))
        self.property_mapping = json.load(open(property_mapping_file_path))
        self.ent_type_mapping = json.load(open(ent_type_mapping_file_path))
        self.entity_mapping_inverse = {
            v: k for k, v in self.entity_mapping.items()}

        self.client = MongoClient(**mongodb_config)
        self.kg_collection = self.client[dbname]['kg']
        self.raw_collection = self.client[dbname]['raw']

        self.topk_aliases = topk_aliases
        self.topk_ent_type = topk_ent_type
        self.langs = langs

    def process_sents(self, sents):
        all_cleaned_texts = []
        all_ents = []
        for block in sents:
            if self.abstract_only and block["paragraph_index"] != 0:
                continue
            if self.eval_extract and block["paragraph_index"] != 1:
                continue
            ents = block['links']
            tokens = block['tokens']
            cleaned_text = ""
            for i, tok in enumerate(tokens):
                if i == 0:
                    cleaned_text += tok['text']
                else:
                    space_num = tokens[i]['start'] - tokens[i-1]['end']
                    cleaned_text += " " * space_num
                    cleaned_text += tok['text']
            all_cleaned_texts.append(cleaned_text)
            all_ents.append(ents)
        return all_cleaned_texts, all_ents

    def process(self, line):
        line = json.loads(line)
        output = {}
        output['id'] = line['id']
        output['title'] = line['title']
        cleaned_text, ents = self.process_sents(line['sentences'])
        output["cleaned_text"] = cleaned_text
        output["ents"] = ents
        output["n_ents"] = len(sum(ents, []))
        return output

    def add_qid(self, record):
        for bid, block in enumerate(record['ents']):
            for eid, ent in enumerate(block):
                name = ent['title']
                try:
                    qid = self.entity_mapping_inverse[name]
                    record['ents'][bid][eid]['qid'] = qid
                except KeyError:
                    continue
        return record

    def collect_qids(self, record):
        ent_map = {}
        for bid, block in enumerate(record['ents']):
            for eid, ent in enumerate(block):
                if 'qid' in ent:
                    ent_map[ent['qid']] = (bid, eid, ent['text'])
        return ent_map

    def add_relations(self, record):
        ent_map = self.collect_qids(record)
        record['n_mapped_ent'] = len(ent_map)

        relations = defaultdict(list)
        for block in record['ents']:
            for ent in block:
                if "qid" in ent:
                    query = {'s': ent["qid"]}
                    res = self.kg_collection.find(query)
                    for item in res:
                        s, p, o = item['s'], item['p'], item['o']
                        if s in ent_map and o in ent_map:
                            relations[(s, o)].append(p)

        augmented_relations = []
        for (s, o), ps in relations.items():
            try:
                relation = {
                    "head": {
                        "mention": ent_map[s][2],
                        "qid": s,
                        "index": (ent_map[s][0], ent_map[s][1])
                    },
                    "tail": {
                        "mention": ent_map[o][2],
                        "qid": o,
                        "index": (ent_map[o][0], ent_map[o][1])
                    },
                    "relations": [{
                        "title": self.property_mapping[p],
                        "pid": p
                    } for p in ps]
                }
                augmented_relations.append(relation)
            except KeyError:
                continue
        record['triplets'] = augmented_relations
        record['n_rel_pair'] = len(record['triplets'])
        record['n_rel'] = sum([len(each['relations'])
                               for each in record['triplets']])
        return record

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

    def add_wikidata_context(self, record):
        # Add entity context
        for sid, block in enumerate(record['ents']):
            for eid, ent in enumerate(block):
                if 'qid' in ent:
                    ent_info = self.collect_wikidata_context(
                        wiki_id=ent['qid'])
                    record['ents'][sid][eid].update(ent_info)

        # Add property context
        for rid, relation in enumerate(record['triplets']):
            for prop_id, property in enumerate(relation['relations']):
                prop_info = self.collect_wikidata_context(
                    wiki_id=property['pid'])
                record['triplets'][rid]['relations'][prop_id].update(prop_info)

        return record

    def add_ent_type(self, record):
        for sid, block in enumerate(record['ents']):
            for eid, ent in enumerate(block):
                if 'qid' in ent and ent['qid'] in self.ent_type_mapping:
                    ent_types = self.ent_type_mapping[ent['qid']]
                    ent_type_names = []
                    for ent_type in ent_types:
                        if ent_type in self.entity_mapping:
                            ent_type_names.append(self.entity_mapping[ent_type])
                    ent_type_names = ent_type_names[:self.topk_ent_type]
                    record['ents'][sid][eid].update(
                            {"type": ent_type_names})
        return record
    
    def add_ent_type_from_kb(self, record):
        '''something wrong'''
        for sid, block in enumerate(record['ents']):
            for eid, ent in enumerate(block):
                if 'qid' in ent:
                    query = {'s': ent['qid'], 'p': 'P31'}
                    res = self.kg_collection.find_one(query)
                    if res and res['o'] in self.entity_mapping:
                        record['ents'][sid][eid].update(
                            {"type": self.entity_mapping[res['o']]})
        return record

    def __call__(self, line):
        record = self.process(line)
        record = self.add_qid(record)
        record = self.add_relations(record)
        record = self.add_wikidata_context(record)
        record = self.add_ent_type(record)
        return record


key_map = {
    "text": "mention",
    "mention": "mention",
    "title": "title",
    "type": "type",
    "description": "description",
    "aliases": "aliases",
}

desire_key_order = ["mention", "title", "type", "description", "aliases"]

def transform(sample, key_map, add_ent_num=False):
    context = ' '.join(sample["cleaned_text"])

    entities = set()
    for ent in sum(sample["ents"], []):
        ent_info = {
                output_name: ent[ori_name]
                for ori_name, output_name in key_map.items() if ori_name in ent and len(ent[ori_name]) != 0
        }
        if 'description' in ent_info:
            ent_info['description'] = ent_info['description']['english']
        if 'aliases' in ent_info:
            ent_info['aliases'] = ent_info['aliases']['english']
        ent_info = OrderedDict([(key, ent_info[key]) for key in desire_key_order if key in ent_info])
        entities.add(json.dumps(ent_info))
    entities = list(map(json.loads, entities))

    relations = set()
    for rel in sample["triplets"]:
        relations.add(
            json.dumps({
                "head": rel["head"]["mention"],
                "tail": rel["tail"]["mention"],
                "relations": [
                    #{
                    #    output_name: p[ori_name]
                    #    for ori_name, output_name in key_map.items() if ori_name in p and len(p[ori_name]) != 0
                    #}
                    p['title']
                    for p in rel["relations"]
                ]
            })
        )
    relations = list(map(json.loads, relations))

    if add_ent_num:
        output = {"number of entities": sample["n_ents"], "entities": entities, "triplets": relations}
    else:
        output = {"entities": entities, "triplets": relations}
    return {"id": sample["id"], "title": sample["title"], "inputs": context, "targets": json.dumps(output)}
