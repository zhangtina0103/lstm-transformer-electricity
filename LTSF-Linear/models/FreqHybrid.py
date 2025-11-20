# # # """
# # # Frequency-Based Hybrid LSTM-Transformer for Time Series Forecasting

# # # Key Innovation:
# # # - Explicit frequency decomposition (FFT-based)
# # # - Role separation: LSTM → trend, Transformer → seasonal, Linear → baseline
# # # - Learnable gating mechanism for adaptive fusion
# # # """

# # # import torch
# # # import torch.nn as nn
# # # import torch.fft

# # # class FreqHybrid(nn.Module):
# # #     def __init__(self, configs):
# # #         super(FreqHybrid, self).__init__()

# # #         # Model configuration
# # #         self.seq_len = configs.seq_len      # Input sequence length L
# # #         self.pred_len = configs.pred_len    # Prediction horizon
# # #         self.enc_in = configs.enc_in        # Number of input features d

# # #         # Frequency band thresholds (tunable hyperparameters)
# # #         self.low_freq_threshold = 0.1   # Bottom 10% = trend (low frequency)
# # #         self.mid_freq_threshold = 0.5   # 10-50% = seasonal (mid frequency)
# # #         # Above 50% = residual/noise (high frequency) - we discard this

# # #         # ====================================================================
# # #         # COMPONENT 1: LSTM Branch (for LOW-FREQUENCY / TREND)
# # #         # ====================================================================
# # #         self.lstm = nn.LSTM(
# # #             input_size=self.enc_in,
# # #             hidden_size=64,
# # #             num_layers=2,
# # #             batch_first=True,
# # #             dropout=0.1
# # #         )
# # #         self.lstm_fc = nn.Linear(64, self.pred_len * self.enc_in)

# # #         # ====================================================================
# # #         # COMPONENT 2: Transformer Branch (for MID-FREQUENCY / SEASONAL)
# # #         # ====================================================================
# # #         self.embedding = nn.Linear(self.enc_in, 128)
# # #         encoder_layer = nn.TransformerEncoderLayer(
# # #             d_model=128,
# # #             nhead=4,
# # #             dim_feedforward=256,
# # #             dropout=0.1,
# # #             batch_first=True
# # #         )
# # #         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
# # #         self.trans_fc = nn.Linear(128, self.pred_len * self.enc_in)

# # #         # ====================================================================
# # #         # COMPONENT 3: Linear Branch (DLinear-style baseline on FULL signal)
# # #         # ====================================================================
# # #         self.linear = nn.Linear(self.seq_len, self.pred_len)

# # #         # ====================================================================
# # #         # COMPONENT 4: Gating Mechanism (learns α1, α2, α3)
# # #         # ====================================================================
# # #         # Projects features from each component to a common space
# # #         self.feature_proj = nn.Linear(self.enc_in, 128)

# # #         # MLP that outputs 3 weights (one per branch)
# # #         self.gate_fc = nn.Sequential(
# # #             nn.Linear(128 * 3, 64),  # Concatenate features from 3 branches
# # #             nn.ReLU(),
# # #             nn.Dropout(0.1),
# # #             nn.Linear(64, 3),
# # #             nn.Softmax(dim=-1)  # α1 + α2 + α3 = 1
# # #         )

# # #     def frequency_decomposition(self, x):
# # #         """
# # #         Decompose input into frequency bands using FFT

# # #         Args:
# # #             x: [batch, seq_len, features] - input time series

# # #         Returns:
# # #             trend: [batch, seq_len, features] - low-frequency component
# # #             seasonal: [batch, seq_len, features] - mid-frequency component
# # #         """
# # #         batch_size, seq_len, n_features = x.shape

# # #         # Apply FFT along time dimension (dim=1)
# # #         x_freq = torch.fft.rfft(x, dim=1)  # [batch, freq_bins, features]

# # #         # Determine frequency cutoffs
# # #         freq_bins = x_freq.shape[1]
# # #         low_cutoff = max(1, int(freq_bins * self.low_freq_threshold))
# # #         mid_cutoff = max(low_cutoff + 1, int(freq_bins * self.mid_freq_threshold))

# # #         # Create binary masks for each frequency band
# # #         low_mask = torch.zeros_like(x_freq)
# # #         mid_mask = torch.zeros_like(x_freq)

# # #         low_mask[:, :low_cutoff, :] = 1.0           # Keep low frequencies
# # #         mid_mask[:, low_cutoff:mid_cutoff, :] = 1.0  # Keep mid frequencies

# # #         # Apply masks and inverse FFT to get time-domain components
# # #         trend = torch.fft.irfft(x_freq * low_mask, n=seq_len, dim=1)
# # #         seasonal = torch.fft.irfft(x_freq * mid_mask, n=seq_len, dim=1)

# # #         return trend, seasonal

# # #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# # #         """
# # #         Forward pass through hybrid architecture

# # #         Args:
# # #             x_enc: [batch, seq_len, features] - input sequence
# # #             x_mark_enc: timestamp features (not used here)
# # #             x_dec: decoder input (not used here)
# # #             x_mark_dec: decoder timestamps (not used here)

# # #         Returns:
# # #             y_pred: [batch, pred_len, features] - final predictions
# # #         """
# # #         batch_size = x_enc.shape[0]

# # #         # ====================================================================
# # #         # STEP 1: Frequency Decomposition
# # #         # ====================================================================
# # #         trend, seasonal = self.frequency_decomposition(x_enc)

# # #         # ====================================================================
# # #         # STEP 2: LSTM Branch (processes TREND)
# # #         # ====================================================================
# # #         lstm_out, (h_n, c_n) = self.lstm(trend)
# # #         lstm_hidden = h_n[-1]  # Use last hidden state [batch, 64]
# # #         y_lstm = self.lstm_fc(lstm_hidden)  # [batch, pred_len * features]
# # #         y_lstm = y_lstm.view(batch_size, self.pred_len, self.enc_in)

# # #         # ====================================================================
# # #         # STEP 3: Transformer Branch (processes SEASONAL)
# # #         # ====================================================================
# # #         trans_in = self.embedding(seasonal)  # [batch, seq_len, 128]
# # #         trans_out = self.transformer(trans_in)  # [batch, seq_len, 128]
# # #         trans_pooled = trans_out.mean(dim=1)  # Global average pooling [batch, 128]
# # #         y_trans = self.trans_fc(trans_pooled)  # [batch, pred_len * features]
# # #         y_trans = y_trans.view(batch_size, self.pred_len, self.enc_in)

# # #         # ====================================================================
# # #         # STEP 4: Linear Branch (processes FULL signal)
# # #         # ====================================================================
# # #         x_lin = x_enc.permute(0, 2, 1)  # [batch, features, seq_len]
# # #         y_lin = self.linear(x_lin)  # [batch, features, pred_len]
# # #         y_lin = y_lin.permute(0, 2, 1)  # [batch, pred_len, features]

# # #         # ====================================================================
# # #         # STEP 5: Gating Mechanism (compute α1, α2, α3)
# # #         # ====================================================================
# # #         # Extract representative features from each component
# # #         feat_trend = self.feature_proj(trend.mean(dim=1))      # [batch, 128]
# # #         feat_seasonal = self.feature_proj(seasonal.mean(dim=1))  # [batch, 128]
# # #         feat_full = self.feature_proj(x_enc.mean(dim=1))       # [batch, 128]

# # #         # Concatenate and pass through MLP
# # #         gate_input = torch.cat([feat_trend, feat_seasonal, feat_full], dim=-1)  # [batch, 384]
# # #         alphas = self.gate_fc(gate_input)  # [batch, 3] where sum=1

# # #         # Reshape for broadcasting
# # #         alpha_trans = alphas[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
# # #         alpha_lstm = alphas[:, 1:2].unsqueeze(-1)   # [batch, 1, 1]
# # #         alpha_lin = alphas[:, 2:3].unsqueeze(-1)    # [batch, 1, 1]

# # #         # ====================================================================
# # #         # STEP 6: Weighted Fusion
# # #         # ====================================================================
# # #         y_pred = alpha_trans * y_trans + alpha_lstm * y_lstm + alpha_lin * y_lin

# # #         return y_pred


# # import torch
# # import torch.nn as nn
# # import torch.fft

# # class Model(nn.Module):
# #     """
# #     Frequency-Based Hybrid LSTM-Transformer for Time Series Forecasting
# #     """
# #     def __init__(self, configs):
# #         super(Model, self).__init__()

# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #         self.enc_in = configs.enc_in

# #         self.low_freq_threshold = 0.1
# #         self.mid_freq_threshold = 0.5

# #         # LSTM for trend
# #         self.lstm = nn.LSTM(self.enc_in, 64, 2, batch_first=True, dropout=0.1)
# #         self.lstm_fc = nn.Linear(64, self.pred_len * self.enc_in)

# #         # Transformer for seasonal
# #         self.embedding = nn.Linear(self.enc_in, 128)
# #         encoder_layer = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True)
# #         self.transformer = nn.TransformerEncoder(encoder_layer, 2)
# #         self.trans_fc = nn.Linear(128, self.pred_len * self.enc_in)

# #         # Linear for baseline
# #         self.linear = nn.Linear(self.seq_len, self.pred_len)

# #         # Gating
# #         self.feature_proj = nn.Linear(self.enc_in, 128)
# #         self.gate_fc = nn.Sequential(
# #             nn.Linear(128 * 3, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(64, 3),
# #             nn.Softmax(dim=-1)
# #         )

# #     def frequency_decomposition(self, x):
# #         """Decompose signal into frequency bands"""
# #         batch_size, seq_len, n_features = x.shape

# #         x_freq = torch.fft.rfft(x, dim=1)

# #         freq_bins = x_freq.shape[1]
# #         low_cutoff = max(1, int(freq_bins * self.low_freq_threshold))
# #         mid_cutoff = max(low_cutoff + 1, int(freq_bins * self.mid_freq_threshold))

# #         low_mask = torch.zeros_like(x_freq)
# #         mid_mask = torch.zeros_like(x_freq)
# #         low_mask[:, :low_cutoff, :] = 1.0
# #         mid_mask[:, low_cutoff:mid_cutoff, :] = 1.0

# #         trend = torch.fft.irfft(x_freq * low_mask, n=seq_len, dim=1)
# #         seasonal = torch.fft.irfft(x_freq * mid_mask, n=seq_len, dim=1)

# #         return trend, seasonal

# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         batch_size = x_enc.shape[0]

# #         # Decompose
# #         trend, seasonal = self.frequency_decomposition(x_enc)

# #         # LSTM branch
# #         lstm_out, (h_n, _) = self.lstm(trend)
# #         y_lstm = self.lstm_fc(h_n[-1]).view(batch_size, self.pred_len, self.enc_in)

# #         # Transformer branch
# #         trans_in = self.embedding(seasonal)
# #         trans_out = self.transformer(trans_in)
# #         y_trans = self.trans_fc(trans_out.mean(dim=1)).view(batch_size, self.pred_len, self.enc_in)

# #         # Linear branch
# #         y_lin = self.linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

# #         # Gating
# #         feat_trend = self.feature_proj(trend.mean(dim=1))
# #         feat_seasonal = self.feature_proj(seasonal.mean(dim=1))
# #         feat_full = self.feature_proj(x_enc.mean(dim=1))
# #         alphas = self.gate_fc(torch.cat([feat_trend, feat_seasonal, feat_full], dim=-1))

# #         # Combine
# #         y_pred = (alphas[:, 0:1, None] * y_trans +
# #                   alphas[:, 1:2, None] * y_lstm +
# #                   alphas[:, 2:3, None] * y_lin)

# #         return y_pred


# """
# Frequency-Based Hybrid LSTM-Transformer for Time Series Forecasting

# Key Innovation:
# - Explicit frequency decomposition (FFT-based)
# - Role separation: LSTM → trend, Transformer → seasonal, Linear → baseline
# - Learnable gating mechanism for adaptive fusion
# """

# import torch
# import torch.nn as nn
# import torch.fft

# class Model(nn.Module):
#     """
#     Frequency-Based Hybrid LSTM-Transformer for Time Series Forecasting
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in

#         # Frequency band thresholds (tunable hyperparameters)
#         self.low_freq_threshold = 0.1   # Bottom 10% = trend (low frequency)
#         self.mid_freq_threshold = 0.5   # 10-50% = seasonal (mid frequency)

#         # ====================================================================
#         # COMPONENT 1: LSTM Branch (for LOW-FREQUENCY / TREND)
#         # ====================================================================
#         self.lstm = nn.LSTM(self.enc_in, 64, 2, batch_first=True, dropout=0.1)
#         self.lstm_fc = nn.Linear(64, self.pred_len * self.enc_in)

#         # ====================================================================
#         # COMPONENT 2: Transformer Branch (for MID-FREQUENCY / SEASONAL)
#         # ====================================================================
#         self.embedding = nn.Linear(self.enc_in, 128)
#         encoder_layer = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, 2)
#         self.trans_fc = nn.Linear(128, self.pred_len * self.enc_in)

#         # ====================================================================
#         # COMPONENT 3: Linear Branch (DLinear-style baseline on FULL signal)
#         # ====================================================================
#         self.linear = nn.Linear(self.seq_len, self.pred_len)

#         # ====================================================================
#         # COMPONENT 4: Gating Mechanism (learns α1, α2, α3)
#         # ====================================================================
#         self.feature_proj = nn.Linear(self.enc_in, 128)
#         self.gate_fc = nn.Sequential(
#             nn.Linear(128 * 3, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, 3),
#             nn.Softmax(dim=-1)  # α1 + α2 + α3 = 1
#         )

#     def frequency_decomposition(self, x):
#         """
#         Decompose input into frequency bands using FFT

#         Args:
#             x: [batch, seq_len, features] - input time series

#         Returns:
#             trend: [batch, seq_len, features] - low-frequency component
#             seasonal: [batch, seq_len, features] - mid-frequency component
#         """
#         batch_size, seq_len, n_features = x.shape

#         # Apply FFT along time dimension
#         x_freq = torch.fft.rfft(x, dim=1)

#         # Determine frequency cutoffs
#         freq_bins = x_freq.shape[1]
#         low_cutoff = max(1, int(freq_bins * self.low_freq_threshold))
#         mid_cutoff = max(low_cutoff + 1, int(freq_bins * self.mid_freq_threshold))

#         # Create binary masks for each frequency band
#         low_mask = torch.zeros_like(x_freq)
#         mid_mask = torch.zeros_like(x_freq)
#         low_mask[:, :low_cutoff, :] = 1.0
#         mid_mask[:, low_cutoff:mid_cutoff, :] = 1.0

#         # Apply masks and inverse FFT to get time-domain components
#         trend = torch.fft.irfft(x_freq * low_mask, n=seq_len, dim=1)
#         seasonal = torch.fft.irfft(x_freq * mid_mask, n=seq_len, dim=1)

#         return trend, seasonal

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         """
#         Forward pass through hybrid architecture

#         Args:
#             x_enc: [batch, seq_len, features] - input sequence

#         Returns:
#             y_pred: [batch, pred_len, features] - final predictions
#         """
#         batch_size = x_enc.shape[0]

#         # STEP 1: Frequency Decomposition
#         trend, seasonal = self.frequency_decomposition(x_enc)

#         # STEP 2: LSTM Branch (processes TREND)
#         lstm_out, (h_n, _) = self.lstm(trend)
#         y_lstm = self.lstm_fc(h_n[-1]).view(batch_size, self.pred_len, self.enc_in)

#         # STEP 3: Transformer Branch (processes SEASONAL)
#         trans_in = self.embedding(seasonal)
#         trans_out = self.transformer(trans_in)
#         y_trans = self.trans_fc(trans_out.mean(dim=1)).view(batch_size, self.pred_len, self.enc_in)

#         # STEP 4: Linear Branch (processes FULL signal)
#         y_lin = self.linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

#         # STEP 5: Gating Mechanism (compute α1, α2, α3)
#         feat_trend = self.feature_proj(trend.mean(dim=1))
#         feat_seasonal = self.feature_proj(seasonal.mean(dim=1))
#         feat_full = self.feature_proj(x_enc.mean(dim=1))
#         alphas = self.gate_fc(torch.cat([feat_trend, feat_seasonal, feat_full], dim=-1))

#         # STEP 6: Weighted Fusion
#         y_pred = (alphas[:, 0:1, None] * y_trans +
#                   alphas[:, 1:2, None] * y_lstm +
#                   alphas[:, 2:3, None] * y_lin)

#         return y_pred


"""
FreqHybrid: Frequency-Aware Hybrid LSTM-Transformer for Electricity Forecasting

Architecture:
- Series decomposition (trend/seasonal split)
- LSTM branch: processes trend (slow-moving patterns)
- Transformer branch: processes seasonal (complex periodic patterns)
- Linear branch: provides stable baseline
- Adaptive gating: learns to weight each branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvg(nn.Module):
    """Moving average block for trend extraction"""
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    Returns: (seasonal, trend)
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    FreqHybrid Model
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Config
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size=25)

        # =====================================================================
        # BRANCH 1: LSTM (for trend - slow-moving patterns)
        # =====================================================================
        self.lstm_hidden = 256
        self.lstm_layers = 3

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=0.1
        )

        # Project LSTM output to predictions
        self.lstm_proj = nn.Sequential(
            nn.Linear(self.lstm_hidden, self.lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.lstm_hidden, self.pred_len * self.enc_in)
        )

        # =====================================================================
        # BRANCH 2: Transformer (for seasonal - complex periodic patterns)
        # =====================================================================
        self.d_model = 512

        # Input embedding
        self.seasonal_embedding = nn.Linear(self.enc_in, self.d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, self.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Output projection
        self.trans_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, self.pred_len * self.enc_in)
        )

        # =====================================================================
        # BRANCH 3: Linear (simple baseline)
        # =====================================================================
        # Separate linears for seasonal and trend
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        # =====================================================================
        # ADAPTIVE GATING MECHANISM
        # =====================================================================
        gate_input_dim = self.lstm_hidden + self.d_model + self.enc_in * 2

        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )

        # Layer norm for stability
        self.norm_lstm = nn.LayerNorm(self.pred_len)
        self.norm_trans = nn.LayerNorm(self.pred_len)
        self.norm_lin = nn.LayerNorm(self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Args:
            x_enc: [batch, seq_len, enc_in]

        Returns:
            output: [batch, pred_len, enc_in]
        """
        batch_size = x_enc.shape[0]

        # =====================================================================
        # STEP 1: Decompose input
        # =====================================================================
        seasonal, trend = self.decomposition(x_enc)

        # =====================================================================
        # STEP 2: LSTM Branch (processes TREND)
        # =====================================================================
        lstm_out, (h_n, c_n) = self.lstm(trend)

        # Use final hidden state
        lstm_hidden = h_n[-1]  # [batch, lstm_hidden]

        # Project to predictions
        y_lstm_flat = self.lstm_proj(lstm_hidden)
        y_lstm = y_lstm_flat.view(batch_size, self.pred_len, self.enc_in)

        # Normalize
        y_lstm = self.norm_lstm(y_lstm.transpose(1, 2)).transpose(1, 2)

        # =====================================================================
        # STEP 3: Transformer Branch (processes SEASONAL)
        # =====================================================================
        # Embed seasonal component
        trans_in = self.seasonal_embedding(seasonal)  # [batch, seq_len, d_model]

        # Add positional encoding
        trans_in = trans_in + self.pos_embedding[:, :self.seq_len, :]

        # Pass through transformer
        trans_out = self.transformer(trans_in)  # [batch, seq_len, d_model]

        # Aggregate: use mean pooling
        trans_agg = trans_out.mean(dim=1)  # [batch, d_model]

        # Project to predictions
        y_trans_flat = self.trans_proj(trans_agg)
        y_trans = y_trans_flat.view(batch_size, self.pred_len, self.enc_in)

        # Normalize
        y_trans = self.norm_trans(y_trans.transpose(1, 2)).transpose(1, 2)

        # =====================================================================
        # STEP 4: Linear Branch (processes BOTH)
        # =====================================================================
        # Linear on seasonal
        y_seasonal_lin = self.linear_seasonal(seasonal.permute(0, 2, 1))
        y_seasonal_lin = y_seasonal_lin.permute(0, 2, 1)

        # Linear on trend
        y_trend_lin = self.linear_trend(trend.permute(0, 2, 1))
        y_trend_lin = y_trend_lin.permute(0, 2, 1)

        # Combine
        y_lin = y_seasonal_lin + y_trend_lin

        # Normalize
        y_lin = self.norm_lin(y_lin.transpose(1, 2)).transpose(1, 2)

        # =====================================================================
        # STEP 5: Adaptive Gating
        # =====================================================================
        # Gather features for gate decision
        gate_features = torch.cat([
            lstm_hidden,              # LSTM final state
            trans_agg,                # Transformer aggregated features
            trend.mean(dim=1),        # Trend summary
            seasonal.mean(dim=1)      # Seasonal summary
        ], dim=-1)

        # Compute gate weights
        alphas = self.gate_network(gate_features)  # [batch, 3]

        # =====================================================================
        # STEP 6: Weighted Combination
        # =====================================================================
        alpha_trans = alphas[:, 0:1, None]  # [batch, 1, 1]
        alpha_lstm = alphas[:, 1:2, None]
        alpha_lin = alphas[:, 2:3, None]

        output = (
            alpha_trans * y_trans +
            alpha_lstm * y_lstm +
            alpha_lin * y_lin
        )

        return output
