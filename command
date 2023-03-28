python -u ./src/run_classifier.py \
	--spiece_model_file=./spiece.model \
	--model_config_path=./xlnet_config.json \
	--init_checkpoint=./xlnet_model.ckpt \
	--task_name=csc \
	--do_train=True \
	--do_eval=False\
	--do_predict=True \
	--eval_all_ckpt=False \
	--uncased=False \
	--data_dir=./ChnSentiCorp \
	--output_dir=./output \
	--model_dir=./model \
	--predict_dir=./result \
	--train_batch_size=48 \
	--eval_batch_size=48 \
	--num_hosts=1 \
	--num_core_per_host=1 \
	--num_train_epochs=12 \
	--max_seq_length=128 \
	--learning_rate=2e-5 \
	--save_steps=5000 \
