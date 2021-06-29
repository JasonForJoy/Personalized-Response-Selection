
CUDA_VISIBLE_DEVICES=0 python -u ../test.py \
    --test_dir ../data_tfrecord/processed_test_self_original.tfrecord \
    --vocab_file ../../uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ../../uncased_L-12_H-768_A-12/bert_config.json \
    --max_seq_length 280 \
    --eval_batch_size 100 \
    --restore_model_dir ../output/PATH_TO_MODEL > log_BERT_NA_test_self_original.txt 2>&1 &
