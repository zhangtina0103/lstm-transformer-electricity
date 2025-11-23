#!/bin/bash

echo "================================"
echo "Weather Forecasting Experiment"
echo "================================"

mkdir -p logs

# Check if weather data exists
if [ ! -f "./dataset/weather.csv" ]; then
    echo "Weather dataset not found!"
    echo "Please download weather.csv to ./dataset/"
    exit 1
fi

echo "Weather dataset found"
echo ""

# =========================================================================
# DLinear Baseline
# =========================================================================
echo "Training DLinear on Weather (Horizon=96)..."
python run_longExp.py \
  --is_training 1 \
  --model_id DLinear_Weather_96_96 \
  --model DLinear \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 5 \
  --use_gpu 1 \
  --gpu 0 \
  --des Baseline \
  2>&1 | tee logs/dlinear_weather_96.log

echo ""
echo "================================"

# =========================================================================
# FreqHybrid
# =========================================================================
echo "Training FreqHybrid on Weather (Horizon=96)..."
python run_longExp.py \
  --is_training 1 \
  --model_id FreqHybrid_Weather_96_96 \
  --model FreqHybrid \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --train_epochs 15 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --patience 5 \
  --use_gpu 1 \
  --gpu 0 \
  --des FreqHybrid \
  2>&1 | tee logs/freqhybrid_weather_96.log

echo ""
echo "================================"
echo "FINAL RESULTS:"
echo "================================"
echo "DLinear:"
grep -E "mse:|mae:" logs/dlinear_weather_96.log | tail -2
echo ""
echo "FreqHybrid:"
grep -E "mse:|mae:" logs/freqhybrid_weather_96.log | tail -2
echo ""
echo "================================"
