from typing import Any, List, Tuple

from pydantic import BaseModel


class BaseResponse(BaseModel):
    query_id: int = None
    total_time_taken: str = None


class GenerateRequest(BaseModel):
    text: List[str] = None
    candidate_mentions: List[str] = None
    min_length: int = 0
    num_beams: int = 1
    do_sample: bool = False
    early_stopping: bool = False
    temperature: float = 1
    top_k: int = 50
    top_p: float = 1
    typical_p: float = 1
    repetition_penalty: float = 1
    bos_token_id: int = None
    pad_token_id: int = None
    eos_token_id: int = None
    length_penalty: float = 1
    no_repeat_ngram_size: int = 0
    encoder_no_repeat_ngram_size: int = 0
    max_time: float = None
    max_new_tokens: int = None
    decoder_start_token_id: int = None
    diversity_penalty: float = 0
    forced_bos_token_id: int = None
    forced_eos_token_id: int = None
    exponential_decay_length_penalty: float = None
    remove_input_from_output: bool = True
    num_return_sequences: int = 1

    def get_generate_kwargs(self) -> dict:
        x = {}
        for k, v in self.dict().items():
            if k not in ["text", "method"] and v is not None:
                x[k] = v
        return x


class GenerateResponse(BaseResponse):
    text: List[str] = None
    num_generated_tokens: List[int] = None
    is_encoder_decoder: bool = False


class TokenizeRequest(BaseModel):
    text: List[str] = None


class TokenizeResponse(BaseResponse):
    token_ids: List[List[int]] = None
    is_encoder_decoder: bool = False


class ForwardRequest(BaseModel):
    conditioning_text: List[str] = None
    response: List[str] = None


class ForwardResponse(BaseResponse):
    nll: float = None
    is_encoder_decoder: bool = False


def parse_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError("{} is not a valid boolean value".format(value))


def parse_field(kwargs: dict, field: str, dtype: type, default_value: Any = None) -> Any:
    if field in kwargs:
        if type(kwargs[field]) == dtype:
            return kwargs[field]
        elif dtype == bool:
            return parse_bool(kwargs[field])
        else:
            return dtype(kwargs[field])
    else:
        return default_value


def create_generate_request(text: List[Tuple[str]], generate_kwargs: dict) -> GenerateRequest:
    # get user generate_kwargs as json and parse it
    return GenerateRequest(
        text=text,
        min_length=parse_field(generate_kwargs, "min_length", int, 0),
        do_sample=parse_field(generate_kwargs, "do_sample", bool, False),
        early_stopping=parse_field(generate_kwargs, "early_stopping", bool, False),
        num_beams=parse_field(generate_kwargs, "num_beams", int, 4),
        temperature=parse_field(generate_kwargs, "temperature", float, 1.0),
        top_k=parse_field(generate_kwargs, "top_k", int, 50),
        top_p=parse_field(generate_kwargs, "top_p", float, 1.0),
        typical_p=parse_field(generate_kwargs, "typical_p", float, 1.0),
        repetition_penalty=parse_field(generate_kwargs, "repetition_penalty", float, 1),
        bos_token_id=parse_field(generate_kwargs, "bos_token_id", int),
        pad_token_id=parse_field(generate_kwargs, "pad_token_id", int),
        eos_token_id=parse_field(generate_kwargs, "eos_token_id", int),
        length_penalty=parse_field(generate_kwargs, "length_penalty", float, 3.0),
        no_repeat_ngram_size=parse_field(generate_kwargs, "no_repeat_ngram_size", int, 0),
        encoder_no_repeat_ngram_size=parse_field(generate_kwargs, "encoder_no_repeat_ngram_size", int, 0),
        max_time=parse_field(generate_kwargs, "max_time", float),
        max_new_tokens=parse_field(generate_kwargs, "max_new_tokens", int, 2048),
        decoder_start_token_id=parse_field(generate_kwargs, "decoder_start_token_id", int),
        num_beam_group=parse_field(generate_kwargs, "num_beam_group", int, 1),
        diversity_penalty=parse_field(generate_kwargs, "diversity_penalty", float, 0.0),
        forced_bos_token_id=parse_field(generate_kwargs, "forced_bos_token_id", int),
        forced_eos_token_id=parse_field(generate_kwargs, "forced_eos_token_id", int),
        exponential_decay_length_penalty=parse_field(generate_kwargs, "exponential_decay_length_penalty", float),
        remove_input_from_output=parse_field(generate_kwargs, "remove_input_from_output", bool, True),
        num_return_sequences=parse_field(generate_kwargs, "num_return_sequences", int, 1),
    )


def get_filter_dict(d: BaseModel) -> dict:
    d = dict(d)
    q = {}
    for i in d:
        if d[i] != None:
            q[i] = d[i]
    del q["text"]
    return q
