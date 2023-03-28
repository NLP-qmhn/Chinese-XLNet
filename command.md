xlnet代码用于在本地重构xlnnet用于下游任务的微调
关于xlnet和bert，差距不是很大，但是两者本地重构有点烦，huggingface提供了远程接口。如果微调模型是基于huggingface 建议忽略xlnet和bert的本地搭建。

我运行的代码如下
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
  配置环境的时候需要注意只能只用tensorflow1.几的版本，我使用的是tensorflow 1.15.0 cuda 10.0 （高版本cuda可能无法和tensorflow1.1适配）
  num_host指的是gpu个数，不过eval和test的使用存在问题，我们现有数据集 test 只有序号和文本，但是代码中将dev数据集当作test来做预测,详见代码378行，
  label=line[1],需要单独做预测时需要修改这里为 label= 0或者1.同时在116行，将eval_split,default改为test。
  如果连续运行两次预测发现结果没有变化，需要清空output文件夹下的tf文件。因为每次预测会先查看，是否已经存在tf文件（用于保存train和test文本特征）如果存在则狐疑直接利用特征文件进行预测。
   
   代码太长建议直接建立 .sh文件 直接运行.sh文件
   

  
