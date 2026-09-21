"""Baseline models compared against DST-TransitNet in the paper (Table 1):
FFNN, LSTM, DLinear, iTransformer.

Reproduction note (compute-budget adaptation, documented in
DST_TRANSITNET_REPRODUCTION_NOTES.md): the paper trains FFNN/LSTM/
iTransformer as 147 independent one-to-one (single-station) models, and
reports the *summed* training time across all 147 models (e.g. 723.8
minutes for LSTM, Table 3). Training 147 fully independent models per
baseline is not tractable within this CPU-only reproduction session. We
instead implement each baseline with the same one-to-one architecture and
per-station inductive bias (no cross-station spatial term, unlike
DST-TransitNet/DLinear), but with parameters *shared* across stations
(the station axis is folded into the batch axis). This preserves the
architecture and the "no spatial information" property the paper compares
against, while making training cost independent of station count. This is
a deviation from strict one-to-one independence and is the single largest
compute-driven simplification in this reproduction -- DST-TransitNet
itself (the paper's actual contribution) is trained exactly as specified,
many-to-many over all stations at once with its full graph/GAT/GRU stack.
DLinear is many-to-many in the paper itself (it shares the decomposition
layer with DST-TransitNet), so no adaptation is needed there.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNNBaseline(nn.Module):
    def __init__(self, recent_len: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = recent_len
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, Xo, Xh, Xt, Xr, adjacency=None):
        B, S, T = Xo.shape
        out = self.net(Xo.reshape(B * S, T)).view(B, S)
        return out


class LSTMBaseline(nn.Module):
    def __init__(self, hidden: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, Xo, Xh, Xt, Xr, adjacency=None):
        B, S, T = Xo.shape
        x = Xo.reshape(B * S, T, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1]).view(B, S)
        return out


class DLinearBaseline(nn.Module):
    """DLinear (Zeng et al., 2023): shares the moving-average temporal
    decomposition with DST-TransitNet, but replaces the GRU/GNN/GAT stack
    with a single linear layer per component (trend, residual), summed.
    Trained many-to-many over all stations at once, exactly as in the paper.
    """

    def __init__(self, recent_len: int):
        super().__init__()
        self.trend_linear = nn.Linear(recent_len, 1)
        self.resid_linear = nn.Linear(recent_len, 1)

    def forward(self, Xo, Xh, Xt, Xr, adjacency=None):
        return self.trend_linear(Xt).squeeze(-1) + self.resid_linear(Xr).squeeze(-1)


class SimpleITransformerBaseline(nn.Module):
    """Simplified iTransformer (Liu et al., 2023): "inverts" the standard
    Transformer by embedding each whole variate series into a single token
    and attending *across variates* rather than across time. Faithfully
    reproducing the full iTransformer (many multivariate covariates,
    dataset-wide attention) trained one-to-one per station as in the paper
    is impractical here; we instead treat the four series available for a
    given station (Xo, Xh, Xt, Xr) as four variate tokens, embed each with a
    shared linear layer, run a small Transformer encoder over these 4
    tokens (self-attention across variates, matching the paper's central
    "inverted" idea), and read out the Xo token's final representation.
    """

    def __init__(self, recent_len: int, hist_len: int, d_model: int = 32, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.embed_o = nn.Linear(recent_len, d_model)
        self.embed_h = nn.Linear(hist_len, d_model)
        self.embed_t = nn.Linear(recent_len, d_model)
        self.embed_r = nn.Linear(recent_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                     dim_feedforward=d_model * 2,
                                                     batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, Xo, Xh, Xt, Xr, adjacency=None):
        B, S, T = Xo.shape
        Xo_, Xh_, Xt_, Xr_ = (t.reshape(B * S, -1) for t in (Xo, Xh, Xt, Xr))
        tok_o = self.embed_o(Xo_)
        tok_h = self.embed_h(Xh_)
        tok_t = self.embed_t(Xt_)
        tok_r = self.embed_r(Xr_)
        tokens = torch.stack([tok_o, tok_h, tok_t, tok_r], dim=1)  # (B*S, 4, d_model)
        enc = self.encoder(tokens)
        out = self.head(enc[:, 0, :]).view(B, S)  # read out the "original series" token
        return out
