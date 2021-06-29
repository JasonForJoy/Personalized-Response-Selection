
CUDA_VISIBLE_DEVICES=0 python -u ../train.py \
  --task_name PersonalizedResponseSelection \
  --train_dir ../data_tfrecord/processed_train_self_original.tfrecord \
  --valid_dir ../data_tfrecord/processed_valid_self_original.tfrecord \
  --output_dir ../output \
  --do_lower_case True \
  --vocab_file ../../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 280 \
  --do_train True  \
  --do_eval True  \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 19 \
  --warmup_proportion 0.1 > log_BERT_NA_train_self_original.txt 2>&1 &
