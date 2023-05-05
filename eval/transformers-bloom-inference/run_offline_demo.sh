CUDA_VISIBLE_DEVICES=5 python -m inference_server.offline_cli_demo \
	--deployment_framework hf_accelerate \
	--model_name /data/home/keminglu/workspace/devcloud/finetune_7b_data_v4_epoch_1 \
	--tokenizer_name /harddisk/user/keminglu/bigscience_tokenizer \
	--model_class AutoModelForCausalLM \
	--dtype fp16\
	--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": false, "num_return_sequences": 1}'
	#--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": true, "num_return_sequences": 10}'
	#--title_trie_path /harddisk/user/keminglu/evaluation_corpus/resources/kilt_titles_trie_dict_bloom.pkl \
