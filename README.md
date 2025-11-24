# Hybrid Frequency Modeling for Time Series Forecasting

This repository contains the implementation and evaluation of FreqHybrid, a frequency-aware hybrid architecture for time series forecasting. FreqHybrid combines LSTM, Transformer, and linear components to handle different frequency components of time series data.

## Overview

FreqHybrid decomposes time series into trend and seasonal components, processing each with specialized architectures: LSTM for low-frequency trends, Transformer for high-frequency seasonal patterns, and a linear branch as a baseline. An adaptive gating mechanism learns to weight each branch's contribution.

Despite its design, FreqHybrid underperforms the DLinear baseline by 15-65% across multiple datasets and forecasting horizons. This project documents the implementation, experiments, and analysis of these results.

## Installation

**Requirements:**

- Python 3.6.9+
- PyTorch 1.9.0
- NumPy, Pandas, Matplotlib, scikit-learn

**Setup:**

```bash
conda create -n LTSF_Linear python=3.6.9
conda activate LTSF_Linear
cd LTSF-Linear
pip install -r requirements.txt
```

**Data:**
Place datasets in `LTSF-Linear/dataset/`. Download benchmark datasets from the [LTSF-Linear repository](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

## Usage

**Run electricity forecasting experiments:**

```bash
cd LTSF-Linear
bash ../experiments/electricity_experiments.sh
```

**Run weather forecasting experiments:**

```bash
cd LTSF-Linear
bash ../experiments/weather_experiments.sh
```

**Train a model:**

```bash
cd LTSF-Linear
python run_longExp.py \
  --is_training 1 \
  --model FreqHybrid \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 321 \
  --train_epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0003
```

## Results

FreqHybrid underperforms DLinear:

- Electricity dataset: 15-25% worse MSE
- Weather dataset: 20-65% worse MSE (varies by horizon)

Identified failure modes:

1. Dataset linearity favors simple models
2. Optimization complexity in hybrid architectures
3. Parameter inefficiency
4. Weak seasonal structure limits frequency decomposition benefits

View results: Open `index.html` in a browser or run `python -m http.server 8000`.

## Architecture

FreqHybrid consists of:

1. Series decomposition (trend/seasonal separation via moving average)
2. LSTM branch (processes trend component)
3. Transformer branch (processes seasonal component)
4. Linear branch (baseline predictions)
5. Adaptive gating (learns branch weights)

Implementation: `FreqHybrid/FreqHybrid.py`

## Project Structure

```
├── FreqHybrid/          # FreqHybrid model implementation
├── LTSF-Linear/         # Benchmark framework (DLinear, Linear, NLinear, etc.)
├── analysis/            # Comparison and analysis scripts
├── experiments/         # Experiment shell scripts
├── images/              # Result visualizations
└── index.html          # Interactive results dashboard
```

## Analysis

Run analysis scripts:

```bash
python analysis/all_comparison.py
python analysis/electricity_comparison.py
python analysis/weather_comparison.py
```

## Acknowledgments

This work builds upon:

- LTSF-Linear (Zeng et al., AAAI 2023)
- FEDformer (Zhou et al., ICML 2022)
- Autoformer (Wu et al., NeurIPS 2021)
- Pyraformer (Liu et al., ICLR 2022)
