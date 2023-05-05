CC=/data/home/keminglu/local_workspace/gcc-8.2/bin/gcc CUDA_VISIBLE_DEVICES=7 python -u -m inference_server.offline_cli_parallel \
	--deployment_framework ds_inference \
	--model_name /data/home/keminglu/workspace/devcloud/finetune_7b_data_v3_step_4000 \
	--tokenizer_name /harddisk/user/keminglu/bigscience_tokenizer \
	--model_class AutoModelForCausalLM \
	--dtype fp16\
	--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": false, "num_return_sequences": 1}'
	#--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": true, "num_return_sequences": 10}'
