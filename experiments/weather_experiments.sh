#!/bin/bash

echo "================================"
echo "Weather: Testing Long Horizons"
echo "================================"

mkdir -p logs

# =========================================================================
# Horizon 336 (14 days)
# =========================================================================
echo ""
echo "Testing Horizon = 336 (14 days)..."
echo "================================"

# DLinear
echo "DLinear (336)..."
python run_longExp.py \
  --is_training 1 \
  --model_id DLinear_Weather_96_336 \
  --model DLinear \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --patience 5 \
  --use_gpu 1 \
  2>&1 | tee logs/dlinear_weather_336.log

# FreqHybrid
echo "FreqHybrid (336)..."
python run_longExp.py \
  --is_training 1 \
  --model_id FreqHybrid_Weather_96_336 \
  --model FreqHybrid \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --patience 5 \
  --use_gpu 1 \
  2>&1 | tee logs/freqhybrid_weather_336.log

# =========================================================================
# Horizon 720 (30 days)
# =========================================================================
echo ""
echo "Testing Horizon = 720 (30 days)..."
echo "================================"

# DLinear
echo "DLinear (720)..."
python run_longExp.py \
  --is_training 1 \
  --model_id DLinear_Weather_96_720 \
  --model DLinear \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --patience 5 \
  --use_gpu 1 \
  2>&1 | tee logs/dlinear_weather_720.log

# FreqHybrid
echo "FreqHybrid (720)..."
python run_longExp.py \
  --is_training 1 \
  --model_id FreqHybrid_Weather_96_720 \
  --model FreqHybrid \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --patience 5 \
  --use_gpu 1 \
  2>&1 | tee logs/freqhybrid_weather_720.log

echo ""
echo "================================"
echo "FINAL COMPARISON"
echo "================================"
echo ""
echo "Horizon 96 (4 days):"
grep "mse:" logs/dlinear_weather_96.log | tail -1
grep "mse:" logs/freqhybrid_weather_96.log | tail -1
echo ""
echo "Horizon 336 (14 days):"
grep "mse:" logs/dlinear_weather_336.log | tail -1
grep "mse:" logs/freqhybrid_weather_336.log | tail -1
echo ""
echo "Horizon 720 (30 days):"
grep "mse:" logs/dlinear_weather_720.log | tail -1
grep "mse:" logs/freqhybrid_weather_720.log | tail -1
