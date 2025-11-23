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
