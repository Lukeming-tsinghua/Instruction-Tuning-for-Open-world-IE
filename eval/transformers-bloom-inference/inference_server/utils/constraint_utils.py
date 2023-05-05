import re
import torch
import pickle
from transformers import StoppingCriteria
from typing import Dict, List
from tqdm import tqdm

try:
    import marisa_trie
except ModuleNotFoundError:
    pass


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters
        self.stop_flag = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        left_cnt = (self.stops[0] == input_ids[0]).sum().item()
        right_cnt_1 = (self.stops[1] == input_ids[0]).sum().item()
        right_cnt_2 = (self.stops[2] == input_ids[0]).sum().item()
        right_cnt_3 = (self.stops[3] == input_ids[0]).sum().item()
        right_cnt_4 = (self.stops[4] == input_ids[0]).sum().item()
        right_cnt_5 = (self.stops[5] == input_ids[0]).sum().item() * 2
        right_cnt = right_cnt_1 + right_cnt_2 + right_cnt_3 + right_cnt_4 + right_cnt_5
        if left_cnt == right_cnt:
            return True

        return False


class StoppingCriteriaCloseJson(StoppingCriteria):

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        left_cnt, right_cnt = 0, 0
        for idx in input_ids[0]:
            token = self.tokenizer.convert_ids_to_tokens(idx.item())
            left_cnt += token.count('{')
            right_cnt += token.count('}')

        if left_cnt == right_cnt:
            return True
        return False


class PrefixConstrainedGeneration:

    def __init__(self,
            tokenizer,
            title_trie_path,
            type_trie_path,
            max_type_num,
            min_type_num,
            force_type=True,
            force_no_triplets=False,
            force_no_description=False,
            force_no_aliases=False,
            verbose=False
        ):

        self.verbose = verbose

        self.tokenizer = tokenizer
        self.token_num = len(self.tokenizer)
        self.max_token_id = self.token_num - 1

        self.force_type = force_type
        self.force_type_pattern = r'{"mention": "[^"]*", "title": "[^"]*",$'

        self.force_no_triplets = force_no_triplets
        self.force_no_triplets_pattern = [r'"triplets":$', r'"triplets": \[$']
        self.force_no_triplets_response = [
            self.tokenizer.encode(' [', add_special_tokens=False),
            self.tokenizer.encode(']}', add_special_tokens=False),
        ]

        self.force_no_description = force_no_description
        self.force_no_description_pattern = [
            r'"description":$',
            r'"description": "n$',
            r'"description": "none"$',
            ]
        self.force_no_description_response = [
            self.tokenizer.encode(' "n', add_special_tokens=False),
            self.tokenizer.encode('one"', add_special_tokens=False),
            self.tokenizer.encode(',', add_special_tokens=False),
        ]

        self.force_no_aliases = force_no_aliases
        self.force_no_aliases_pattern = [r'"aliases":$', r'"aliases": \[$']
        self.force_no_aliases_response = [
            self.tokenizer.encode(' [', add_special_tokens=False),
            self.tokenizer.encode(']}', add_special_tokens=False),
        ]

        self.title_trie_path = title_trie_path
        if self.title_trie_path:
            self.title_trie = Trie.load_from_dict(pickle.load(open(self.title_trie_path, "rb")))
            self.title_constraint = True
        else:
            self.title_constraint = False

        self.title_constrained_activate_token = 243001
        self.title_constrained_start_idx = None

        self.type_trie_path = type_trie_path
        if self.type_trie_path:
            self.type_trie = Trie.load_from_dict(pickle.load(open(self.type_trie_path, "rb")))
            open_terms, closed_terms = [], []
            for term in self.type_trie:
                if self.tokenizer.convert_ids_to_tokens([term[-1]])[0][-1] == ']':
                    closed_terms.append(term)
                else:
                    open_terms.append(term)
            self.type_trie_open = Trie(open_terms)
            self.type_trie_closed = Trie(closed_terms)
            self.max_type_num = max_type_num
            self.min_type_num = min_type_num
            self.type_constraint = True
        else:
            self.type_constraint = False

        self.type_constrained_activate_token = 116220
        self.type_constrained_start_idx = None
        self.generated_type_num = 0
    
    def __call__(self, batch_id, sents):
        if self.verbose:
            string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sents))
            print(string)
            #print([(id.item(), token) for id, token in zip(sents, self.tokenizer.convert_ids_to_tokens(sents))])

        if self.title_constraint and sents[-1] == self.title_constrained_activate_token:
            self.title_constrained_start_idx = len(sents) - 1

        if self.title_constrained_start_idx:
            constrained_prefix = [each.item() for each in sents[self.title_constrained_start_idx:]]
            next_tokens = self.title_trie.get(constrained_prefix)
            if len(next_tokens) == 0:
                self.title_constrained_start_idx = None
            else:
                return next_tokens
        
        string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(sents))

        if self.force_no_triplets:
            if re.search(self.force_no_triplets_pattern[0], string):
                return self.force_no_triplets_response[0]
            if re.search(self.force_no_triplets_pattern[1], string):
                return self.force_no_triplets_response[1]

        if self.force_no_description:
            if re.search(self.force_no_description_pattern[0], string):
                return self.force_no_description_response[0]
            if re.search(self.force_no_description_pattern[1], string):
                return self.force_no_description_response[1]
            if re.search(self.force_no_description_pattern[2], string):
                return self.force_no_description_response[2]

        if self.force_no_aliases:
            if re.search(self.force_no_aliases_pattern[0], string):
                return self.force_no_aliases_response[0]
            if re.search(self.force_no_aliases_pattern[1], string):
                return self.force_no_aliases_response[1]

        if self.force_type:
            if re.search(self.force_type_pattern, string):
                return [self.type_constrained_activate_token]

        if self.type_constraint and sents[-1] == self.type_constrained_activate_token:
            self.type_constrained_start_idx = len(sents) - 1
        
        if self.type_constrained_start_idx:
            constrained_prefix = [each.item() for each in sents[self.type_constrained_start_idx:]]

            if self.generated_type_num < self.min_type_num:
                cur_type_trie = self.type_trie_open
            elif self.max_type_num and self.generated_type_num >= self.max_type_num - 1:
                cur_type_trie = self.type_trie_closed
            else:
                cur_type_trie = self.type_trie
            
            next_tokens = cur_type_trie.get(constrained_prefix)
            
            if len(next_tokens) == 0:
                self.generated_type_num += 1
                last_token = constrained_prefix[-1]
                if last_token in cur_type_trie.trie_dict:
                    constrained_prefix = [last_token]
                    next_tokens = cur_type_trie.get(constrained_prefix)
                    self.type_constrained_start_idx = len(sents) - 1
                    return next_tokens
                else:
                    self.type_constrained_start_idx = None
                    self.generated_type_num = 0
                    return range(self.token_num)
            return next_tokens

        return range(self.token_num)


class Trie(object):

    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            pbar = tqdm(total=len(sequences))
            pbar.set_description("building trie:")
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
                pbar.update(1)

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict,
                                   self.append_trie, self.bos_token_id)

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):

        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(prefix_sequence + [next_token],
                                         trie_dict[next_token])
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class MarisaTrie(object):

    def __init__(
        self,
        sequences: List[List[int]] = [],
        cache_fist_branch=True,
        max_token_id=256001,
    ):

        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id +
                                   10000)] if max_token_id >= 55000 else [])
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence])
            for sequence in sequences)

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (self.cache_fist_branch and len(prefix_sequence) == 1
              and self.zero_iter == prefix_sequence):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list({
                self.char2int[e[len(key)]]
                for e in self.trie.keys(key) if len(e) > len(key)
            })

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)


class DummyTrieMention(object):

    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class DummyTrieEntity(object):

    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(
                set(codes[e] for e in (
                    "start_mention_token",
                    "end_mention_token",
                    "start_entity_token",
                ))))
        self._codes = codes

    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["end_mention_token"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["start_entity_token"]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices
                 ) == 1 and indices[0] == self._codes["end_entity_token"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)
