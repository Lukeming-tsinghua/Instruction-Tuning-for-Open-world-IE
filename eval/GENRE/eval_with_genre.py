from genre.hf_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
import os
import json
import torch
from multiprocessing import Pool
from tqdm import trange
import numpy as np


def inference(
    node_data,
    batch_size=4,
    config="./models/hf_e2e_entity_linking_aidayago",
):
    num_batch = int(np.ceil(len(node_data) / batch_size))

    device = torch.device(f"cuda:7")
    model = GENRE.from_pretrained(config).eval()
    model = model.to(device)

    results = []
    for i in trange(num_batch):
        batch_begin, batch_end = i * batch_size, (i+1) * batch_size
        batch_data = node_data[batch_begin:batch_end]
        text_inputs = [each['inputs'] for each in batch_data]
        try:
            outputs = get_entity_spans(
                model,
                text_inputs
            )
            assert len(batch_data) == len(outputs)
            for k in range(len(outputs)):
                try:
                    processed_output = [(item[3], item[2].replace("_", " ")) for item in outputs[k] if len(item) == 4]
                except IndexError:
                    print(outputs[k])
                batch_data[k]['outputs'] = processed_output
            results.extend(batch_data)
        except Exception as e:
            print(e)
            pass
    return results


if __name__ == "__main__":
    with open("/harddisk/user/keminglu/pretrained_data_processed/wikipedia_with_mention_wo_title_simplified_aug_eval/corpus_filtered_test_prompt_rephrased") as f:
        data = [json.loads(line) for line in f.readlines()]
    data = [sample for sample in data if sample['aug_type'] == 'aug_default'][15000:20000]

    results = inference(data)

    with open("genre_output_3.json", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")