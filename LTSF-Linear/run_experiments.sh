#!/bin/bash

# Create logs directory
mkdir -p logs

echo "Starting experiments..."

# DLinear baseline - Horizon 24
echo "Running DLinear (horizon=24)..."
python run_longExp.py \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 5 \
  --batch_size 32 \
  --itr 1 \
  2>&1 | tee logs/dlinear_h24.log

# DLinear baseline - Horizon 96
echo "Running DLinear (horizon=96)..."
python run_longExp.py \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 5 \
  --batch_size 32 \
  --itr 1 \
  2>&1 | tee logs/dlinear_h96.log

# FreqHybrid - Horizon 24
echo "Running FreqHybrid (horizon=24)..."
python run_longExp.py \
  --model FreqHybrid \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 5 \
  --batch_size 32 \
  --itr 1 \
  2>&1 | tee logs/hybrid_h24.log

# FreqHybrid - Horizon 96
echo "Running FreqHybrid (horizon=96)..."
python run_longExp.py \
  --model FreqHybrid \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 5 \
  --batch_size 32 \
  --itr 1 \
  2>&1 | tee logs/hybrid_h96.log

echo "✅ All experiments complete!"
echo "Check results in logs/ directory"
