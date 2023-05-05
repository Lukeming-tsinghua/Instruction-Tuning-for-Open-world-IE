import argparse
import json
import sys
import os
import numpy as np
from tqdm import trange
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .model_handler import ModelDeployment
from .utils import get_argument_parser, parse_args, print_rank_0, collect_precalculate_entities
from .utils import write_file_rank_0
from .utils import get_world_size


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()
    args = parse_args(parser)
    return args


def main() -> None:
    args = get_args()

    model = ModelDeployment(args, True)

    generate_kwargs = args.generate_kwargs

    node_rank = os.environ['INDEX']
    node_size = os.environ['HOST_NUM']
    local_rank = args.gpu_rank
    world_size = node_rank * node_size * 8
    current_rank = node_rank * 8 + local_rank
    total_data_amount = 65532
    length =int(np.ceil(total_data_amount / world_size))
    start, end = current_rank * length, (current_rank + 1) * length


    data_dir = "/data/home/keminglu/workspace/evaluation_corpus/wiki_aug_eval"
    input_file = "corpus_filtered"
    with open(os.path.join(data_dir, input_file)) as f:
        inputs = [json.loads(line.strip()) for line in f.readlines()][:1000]
    
    responses = []
    for i in trange(len(inputs)):
        input_text = inputs[i]['inputs']
        response = model.generate(text=[input_text], generate_kwargs=generate_kwargs)
        responses.extend(response.text)
    
    output_path = os.path.join(data_dir, args.model_name.split("/")[-1] + "_" + input_file + "_output.txt")
    with open(output_path, "w") as f:
        for input, response in zip(inputs, responses):
            try:
                input['output'] = json.dumps(response)
                input['output'] = json.loads(input['output'])
            except json.JSONDecodeError:
                input['output'] = None
            input['targets'] = json.loads(input['targets'])
            f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    main()
