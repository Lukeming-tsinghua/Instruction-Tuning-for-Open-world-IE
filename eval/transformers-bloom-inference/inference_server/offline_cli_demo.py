import argparse
import json
import sys

from .model_handler import ModelDeployment
from .utils import get_argument_parser, parse_args, print_rank_0, collect_precalculate_entities


def get_args() -> argparse.Namespace:
    parser = get_argument_parser()
    args = parse_args(parser)
    return args


def main() -> None:
    args = get_args()

    model = ModelDeployment(args, True)

    generate_kwargs = args.generate_kwargs

    while True:
        input_text = input("Input text: ").strip()
        prompt = input("Prompt input: ").strip()

        response = model.generate(text=[input_text + " "+ prompt], generate_kwargs=generate_kwargs)

        #precal_entities = collect_precalculate_entities(response.text)
        #print_rank_0(precal_entities)

        for text, num in zip(response.text, response.num_generated_tokens):
            print_rank_0("Output text:", prompt + text)
            print_rank_0("Generated tokens:", num)


if __name__ == "__main__":
    main()
