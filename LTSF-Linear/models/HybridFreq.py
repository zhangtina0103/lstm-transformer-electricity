"""
Frequency-Based Hybrid LSTM-Transformer for Time Series Forecasting

Key Innovation:
- Explicit frequency decomposition (FFT-based)
- Role separation: LSTM → trend, Transformer → seasonal, Linear → baseline
- Learnable gating mechanism for adaptive fusion
"""

import torch
import torch.nn as nn
import torch.fft

class FreqHybrid(nn.Module):
    def __init__(self, configs):
        super(FreqHybrid, self).__init__()

        # Model configuration
        self.seq_len = configs.seq_len      # Input sequence length L
        self.pred_len = configs.pred_len    # Prediction horizon
        self.enc_in = configs.enc_in        # Number of input features d

        # Frequency band thresholds (tunable hyperparameters)
        self.low_freq_threshold = 0.1   # Bottom 10% = trend (low frequency)
        self.mid_freq_threshold = 0.5   # 10-50% = seasonal (mid frequency)
        # Above 50% = residual/noise (high frequency) - we discard this

        # ====================================================================
        # COMPONENT 1: LSTM Branch (for LOW-FREQUENCY / TREND)
        # ====================================================================
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.lstm_fc = nn.Linear(64, self.pred_len * self.enc_in)

        # ====================================================================
        # COMPONENT 2: Transformer Branch (for MID-FREQUENCY / SEASONAL)
        # ====================================================================
        self.embedding = nn.Linear(self.enc_in, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.trans_fc = nn.Linear(128, self.pred_len * self.enc_in)

        # ====================================================================
        # COMPONENT 3: Linear Branch (DLinear-style baseline on FULL signal)
        # ====================================================================
        self.linear = nn.Linear(self.seq_len, self.pred_len)

        # ====================================================================
        # COMPONENT 4: Gating Mechanism (learns α1, α2, α3)
        # ====================================================================
        # Projects features from each component to a common space
        self.feature_proj = nn.Linear(self.enc_in, 128)

        # MLP that outputs 3 weights (one per branch)
        self.gate_fc = nn.Sequential(
            nn.Linear(128 * 3, 64),  # Concatenate features from 3 branches
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)  # α1 + α2 + α3 = 1
        )

    def frequency_decomposition(self, x):
        """
        Decompose input into frequency bands using FFT

        Args:
            x: [batch, seq_len, features] - input time series

        Returns:
            trend: [batch, seq_len, features] - low-frequency component
            seasonal: [batch, seq_len, features] - mid-frequency component
        """
        batch_size, seq_len, n_features = x.shape

        # Apply FFT along time dimension (dim=1)
        x_freq = torch.fft.rfft(x, dim=1)  # [batch, freq_bins, features]

        # Determine frequency cutoffs
        freq_bins = x_freq.shape[1]
        low_cutoff = max(1, int(freq_bins * self.low_freq_threshold))
        mid_cutoff = max(low_cutoff + 1, int(freq_bins * self.mid_freq_threshold))

        # Create binary masks for each frequency band
        low_mask = torch.zeros_like(x_freq)
        mid_mask = torch.zeros_like(x_freq)

        low_mask[:, :low_cutoff, :] = 1.0           # Keep low frequencies
        mid_mask[:, low_cutoff:mid_cutoff, :] = 1.0  # Keep mid frequencies

        # Apply masks and inverse FFT to get time-domain components
        trend = torch.fft.irfft(x_freq * low_mask, n=seq_len, dim=1)
        seasonal = torch.fft.irfft(x_freq * mid_mask, n=seq_len, dim=1)

        return trend, seasonal

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass through hybrid architecture

        Args:
            x_enc: [batch, seq_len, features] - input sequence
            x_mark_enc: timestamp features (not used here)
            x_dec: decoder input (not used here)
            x_mark_dec: decoder timestamps (not used here)

        Returns:
            y_pred: [batch, pred_len, features] - final predictions
        """
        batch_size = x_enc.shape[0]

        # ====================================================================
        # STEP 1: Frequency Decomposition
        # ====================================================================
        trend, seasonal = self.frequency_decomposition(x_enc)

        # ====================================================================
        # STEP 2: LSTM Branch (processes TREND)
        # ====================================================================
        lstm_out, (h_n, c_n) = self.lstm(trend)
        lstm_hidden = h_n[-1]  # Use last hidden state [batch, 64]
        y_lstm = self.lstm_fc(lstm_hidden)  # [batch, pred_len * features]
        y_lstm = y_lstm.view(batch_size, self.pred_len, self.enc_in)

        # ====================================================================
        # STEP 3: Transformer Branch (processes SEASONAL)
        # ====================================================================
        trans_in = self.embedding(seasonal)  # [batch, seq_len, 128]
        trans_out = self.transformer(trans_in)  # [batch, seq_len, 128]
        trans_pooled = trans_out.mean(dim=1)  # Global average pooling [batch, 128]
        y_trans = self.trans_fc(trans_pooled)  # [batch, pred_len * features]
        y_trans = y_trans.view(batch_size, self.pred_len, self.enc_in)

        # ====================================================================
        # STEP 4: Linear Branch (processes FULL signal)
        # ====================================================================
        x_lin = x_enc.permute(0, 2, 1)  # [batch, features, seq_len]
        y_lin = self.linear(x_lin)  # [batch, features, pred_len]
        y_lin = y_lin.permute(0, 2, 1)  # [batch, pred_len, features]

        # ====================================================================
        # STEP 5: Gating Mechanism (compute α1, α2, α3)
        # ====================================================================
        # Extract representative features from each component
        feat_trend = self.feature_proj(trend.mean(dim=1))      # [batch, 128]
        feat_seasonal = self.feature_proj(seasonal.mean(dim=1))  # [batch, 128]
        feat_full = self.feature_proj(x_enc.mean(dim=1))       # [batch, 128]

        # Concatenate and pass through MLP
        gate_input = torch.cat([feat_trend, feat_seasonal, feat_full], dim=-1)  # [batch, 384]
        alphas = self.gate_fc(gate_input)  # [batch, 3] where sum=1

        # Reshape for broadcasting
        alpha_trans = alphas[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
        alpha_lstm = alphas[:, 1:2].unsqueeze(-1)   # [batch, 1, 1]
        alpha_lin = alphas[:, 2:3].unsqueeze(-1)    # [batch, 1, 1]

        # ====================================================================
        # STEP 6: Weighted Fusion
        # ====================================================================
        y_pred = alpha_trans * y_trans + alpha_lstm * y_lstm + alpha_lin * y_lin

        return y_pred
