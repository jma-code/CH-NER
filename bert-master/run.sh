export BERT_BASE_DIR=../bert/chinese_L-12_H-768_A-12
export NER_DIR=../data/tmp
python3 run_NER.py \
          --task_name=NER \
          --do_train=true \
          --do_eval=true \
          --do_predict=true \
          --data_dir=$NER_DIR/ \
          --vocab_file=$BERT_BASE_DIR/vocab.txt \
          --bert_config_file=$BERT_BASE_DIR/bert_config.json \
          --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
          --max_seq_length=256 \
          --train_batch_size=32 \
          --learning_rate=2e-5 \
          --num_train_epochs=5.0 \
          --output_dir=$BERT_BASE_DIR/output/
