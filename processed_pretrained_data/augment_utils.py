import json
import copy
import random


def augmentation(func):
    def wrapper(record, **kwargs):
        record = copy.deepcopy(record)
        targets = json.loads(record['targets'])
        entities = targets['entities']
        triplets = targets['triplets']
        aug_sample = func(entities, triplets, **kwargs)

        if not aug_sample:
            return

        new_entities, new_triplets, prompt, aug_infos = aug_sample
        targets['entities'] = new_entities
        targets['triplets'] = new_triplets
        record['targets'] = json.dumps(targets)
        record['inputs'] = record['inputs']
        record['prompt'] = prompt
        record['aug_type'] = func.__name__
        record['aug_info'] = aug_infos
        return record

    return wrapper

@augmentation
def aug_default(entities, triplets):
    prompt = "Extract entities."
    return entities, triplets, prompt, None

@augmentation
def aug_ent_num(entities, triplets, sampling=False):
    ent_num = len(entities)
    if sampling:
        ent_num = random.randint(1, len(entities))
        new_entities = entities[:ent_num]
        mentions = set(each['mention'] for each in new_entities)

        new_triplets = []
        for triplet in triplets:
            if triplet['head'] in mentions and triplet['tail'] in mentions:
                new_triplets.append(triplet)

        entities = new_entities
        triplets = new_triplets

    prompt = f"Extract {ent_num} entities." if ent_num > 1 else f"Extract {ent_num} entity."
    return entities, triplets, prompt, ent_num

@augmentation
def aug_importance(entities, triplets, prior_map):

    entities = [(each, prior_map[each['title']]) for each in entities if each['title'] in prior_map]
    entities = sorted(entities, key=lambda x: -x[1])

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    new_entities = [each[0] for each in entities[:ent_num]]
    mentions = set(each['mention'] for each in new_entities)

    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    prompt = f"Extract the most important {ent_num} entities." if ent_num > 1 else f"Extract the most important entity."
    return new_entities, new_triplets, prompt, ent_num

@augmentation
def aug_base_type(entities, triplets):
    types = [(each, each['type']) for each in entities if 'type' in each]

    if not types:
        return

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    random.shuffle(types)
    types = types[:ent_num]
    new_entities = [each[0] for each in types]
    candidate_types = [random.choice(each[1]) for each in types]
    type_str = ", ".join(set(candidate_types))

    mentions = set(each['mention'] for each in new_entities)
    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    prompt = f"Extract entities in types {type_str}." if len(set(candidate_types)) != 1 \
        else f"Extract entities in the type {type_str}" 
    return new_entities, new_triplets, prompt, type_str

@augmentation
def aug_rollup_type(entities, triplets, label2qid, qid2label, ontology):
    types = []
    for each in entities:
        if 'type' in each:
            rollup_types = []
            for item in each['type']:
                if item in label2qid and label2qid[item] in ontology:
                    rollup_type_id = ontology[label2qid[item]]
                    rollup_type = [qid2label[qid] for qid in rollup_type_id if qid in qid2label]
                    rollup_types += rollup_type
            rollup_types = list(set(rollup_types))
            if rollup_types:
                types.append((each, rollup_types))

    if not types:
        return

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    random.shuffle(types)
    types = types[:ent_num]
    new_entities, candidate_types = [], []
    for ent, type in types:
        choice_type = random.choice(type)
        ent['type'].append(choice_type)
        new_entities.append(ent)
        candidate_types.append(choice_type)
    type_str = ", ".join(set(candidate_types))

    mentions = set(each['mention'] for each in new_entities)
    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    prompt = f"Extract entities in types {type_str}." if len(set(candidate_types)) != 1 \
        else f"Extract entities in the type {type_str}"
    return new_entities, new_triplets, prompt, type_str


@augmentation
def aug_ent_num_and_base_type(entities, triplets):
    types = [(each, each['type']) for each in entities if 'type' in each]

    if not types:
        return

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    random.shuffle(types)
    types = types[:ent_num]
    new_entities = [each[0] for each in types]
    candidate_types = [random.choice(each[1]) for each in types]
    type_str = ", ".join(set(candidate_types))

    mentions = set(each['mention'] for each in new_entities)
    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    # whether singular
    prompts = {
        (True, True): f"Extract {ent_num} entity in the type {type_str}.",
        (True, False): f"Extract {ent_num} entity in types {type_str}.",
        (False, True): f"Extract {ent_num} entities in the type {type_str}.",
        (False, False): f"Extract {ent_num} entities in types {type_str}.",
    }
    prompt = prompts[(ent_num == 1, len(set(candidate_types)) == 1)]
    return new_entities, new_triplets, prompt, [ent_num, type_str]

@augmentation
def aug_ent_num_and_rollup_type(entities, triplets, label2qid, qid2label, ontology):
    types = []
    for each in entities:
        if 'type' in each:
            rollup_types = []
            for item in each['type']:
                if item in label2qid and label2qid[item] in ontology:
                    rollup_type_id = ontology[label2qid[item]]
                    rollup_type = [qid2label[qid] for qid in rollup_type_id if qid in qid2label]
                    rollup_types += rollup_type
            rollup_types = list(set(rollup_types))
            if rollup_types:
                types.append((each, rollup_types))

    if not types:
        return

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    random.shuffle(types)
    types = types[:ent_num]
    new_entities, candidate_types = [], []
    for ent, type in types:
        choice_type = random.choice(type)
        ent['type'].append(choice_type)
        new_entities.append(ent)
        candidate_types.append(choice_type)
    type_str = ", ".join(set(candidate_types))

    mentions = set(each['mention'] for each in new_entities)
    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    # whether singular
    prompts = {
        (True, True): f"Extract {ent_num} entity in the type {type_str}.",
        (True, False): f"Extract {ent_num} entity in types {type_str}.",
        (False, True): f"Extract {ent_num} entities in the type {type_str}.",
        (False, False): f"Extract {ent_num} entities in types {type_str}.",
    }
    prompt = prompts[(ent_num == 1, len(set(candidate_types)) == 1)]
    return new_entities, new_triplets, prompt, [ent_num, type_str]


@augmentation
def aug_description(entities, triplets):
    descriptions = [(each, each['description']) for each in entities if 'description' in each]

    if not descriptions:
        return

    ent_num = 0 if len(entities) == 0 else random.randint(1, len(entities))
    if ent_num == 0:
        return
    random.shuffle(descriptions)
    descriptions = descriptions[:ent_num]
    new_entities = [each[0] for each in descriptions]
    descriptions = [each[1] for each in descriptions]
    description_str = "; ".join(descriptions)

    mentions = set(each['mention'] for each in new_entities)
    new_triplets = []
    for triplet in triplets:
        if triplet['head'] in mentions and triplet['tail'] in mentions:
            new_triplets.append(triplet)

    prompt = f"Extract entities in following descriptions: {description_str}." if len(descriptions) > 1\
        else f"Extract entities in the following description: {description_str}"
    return new_entities, new_triplets, prompt, description_str


def augment_sample(sample,
        prior_map,     
        label2qid,
        qid2label,
        ontology,
        eval_split=False,
        logger=None
    ):

    funcs = [
        (aug_ent_num, {"sampling": True}),
        (aug_importance, {"prior_map": prior_map}),
        (aug_base_type, {}),
        (aug_rollup_type, {
            "label2qid": label2qid,
            "qid2label": qid2label,
            "ontology": ontology
        }),
        (aug_description, {}),
    ]

    if eval_split:
        funcs.append((aug_ent_num_and_base_type, {}))
        funcs.append((aug_ent_num_and_rollup_type, {
            "label2qid": label2qid,
            "qid2label": qid2label,
            "ontology": ontology
        }))

    outputs = [aug_default(sample)]
    try:
        augmented_outputs = []
        for func, kwargs in funcs:
            result = func(sample, **kwargs)
            if result:
                augmented_outputs.append(result)
        random.shuffle(augmented_outputs)
        if augmented_outputs:
            outputs.append(augmented_outputs[0])
    except Exception as e:
        print(e)
    return outputs