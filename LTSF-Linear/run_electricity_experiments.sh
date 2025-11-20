#!/bin/bash

echo "================================"
echo "Electricity Load Forecasting"
echo "================================"

mkdir -p logs

# Configuration
DATA="custom"
DATA_PATH="electricity.csv"
ROOT_PATH="./dataset/"
SEQ_LEN=96
PRED_LEN=96
ENC_IN=321
BATCH_SIZE=16
TRAIN_EPOCHS=15

echo "Training DLinear baseline..."
python run_longExp.py \
  --is_training 1 \
  --model_id DLinear_ECL_96_96 \
  --model DLinear \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len 48 \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${ENC_IN} \
  --c_out ${ENC_IN} \
  --des Baseline \
  --train_epochs ${TRAIN_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate 0.0001 \
  --itr 1 \
  2>&1 | tee logs/dlinear_ecl_96.log

echo ""
echo "Training FreqHybrid model..."
python run_longExp.py \
  --is_training 1 \
  --model_id FreqHybrid_ECL_96_96 \
  --model FreqHybrid \
  --data ${DATA} \
  --root_path ${ROOT_PATH} \
  --data_path ${DATA_PATH} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len 48 \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${ENC_IN} \
  --c_out ${ENC_IN} \
  --des FreqHybrid \
  --train_epochs ${TRAIN_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate 0.0003 \
  --itr 1 \
  2>&1 | tee logs/freqhybrid_ecl_96.log

echo ""
echo "================================"
echo "RESULTS:"
echo "================================"
echo "DLinear:"
grep -E "mse:|mae:" logs/dlinear_ecl_96.log | tail -2
echo ""
echo "FreqHybrid:"
grep -E "mse:|mae:" logs/freqhybrid_ecl_96.log | tail -2
