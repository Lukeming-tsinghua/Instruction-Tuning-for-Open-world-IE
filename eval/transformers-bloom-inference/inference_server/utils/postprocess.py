import json
from collections import defaultdict

def collect_precalculate_entities(outputs):
    format_corrupt_cnt = 0
    parsed_outputs = []
    for output in outputs:
        try:
            parsed_outputs.append(json.loads(output))
        except json.JSONDecodeError:
            format_corrupt_cnt += 1
    
    entities = defaultdict(lambda: 0)
    for output in parsed_outputs:
        for ent in output["entities"]:
            entities[(ent["mention"], ent["title"])] += 1
    
    for key in entities:
        entities[key] /= len(outputs)

    return entities