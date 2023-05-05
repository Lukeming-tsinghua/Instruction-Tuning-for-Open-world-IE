import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

accelerator = Accelerator()


data_config = {
    "data_dir": "/harddisk/user/keminglu/evaluation_corpus/",
    "data_file": "wiki_eval.txt",
}

def get_dataloader(data_dir, data_file, tokenizer):
    dataset = load_dataset(path=data_dir, data_files=data_file)["train"].select(range(10))

    def tokenize_function(example):
        inputs = tokenizer(example["text"])
        return inputs

    def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")["input_ids"]

    tokenized_datasets = dataset.map(
                tokenize_function,
                remove_columns=["text"],
            )

    dataloader = DataLoader(tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=1)
    return dataloader

tokenizer = AutoTokenizer.from_pretrained("/harddisk/user/keminglu/bigscience_tokenizer")
model = AutoModelForCausalLM.from_pretrained("/data/home/keminglu/workspace/dev-cloud/finetune_1b1_data_v1_epoch_1")

dataloader = get_dataloader(**data_config, tokenizer=tokenizer)

dataloader = accelerator.prepare(dataloader)

model = model.to(accelerator.device)
model.eval()

all_results = []
pbar = tqdm(total=len(dataloader))
for inputs in dataloader:

    with torch.no_grad():
        outputs = model.generate(
                inputs=inputs,
                num_beams=4,
                length_penalty=1,
                do_sample=False,
                max_new_tokens=2048
            )

    prompt_length = len(
                tokenizer.decode(
                    inputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    texts = [text[prompt_length:].strip() for text in texts]

    all_results.append(texts)
    pbar.update(1)

print(len(all_results))
