CUDA_VISIBLE_DEVICES=4,5,6,7 python -m inference_server.cli \
	--deployment_framework hf_accelerate\
	--model_name /data/home/keminglu/workspace/devcloud/finetune_7b_data_v1_epoch_1 \
	--tokenizer_name /harddisk/user/keminglu/bigscience_tokenizer \
	--model_class AutoModelForCausalLM \
	--dtype fp16\
	--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": false}'
