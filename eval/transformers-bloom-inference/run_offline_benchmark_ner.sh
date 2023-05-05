task=ner
dataset=conllpp_gold
CUDA_VISIBLE_DEVICES=4 nohup python -u -m inference_server.offline_cli_benchmark \
	--deployment_framework ds_inference \
	--model_name /data/home/keminglu/workspace/devcloud/finetune_7b_data_v4_epoch_1 \
	--tokenizer_name /harddisk/user/keminglu/bigscience_tokenizer \
	--force_type \
	--max_type_num 1\
	--min_type_num 0 \
	--model_class AutoModelForCausalLM \
	--dtype fp16\
	--task $task \
	--dataset $dataset \
	--save_results \
	--generate_kwargs '{"num_beams": 1, "max_new_tokens": 2048, "do_sample": false, "num_return_sequences": 1}' >> logs/${task}_${dataset}_7b_v4.log &
	#--generate_kwargs '{"num_beams": 4, "max_new_tokens": 2048, "do_sample": true, "num_return_sequences": 10}'
