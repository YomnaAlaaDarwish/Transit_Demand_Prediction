"""DST-TransitNet and DST-TransitNetV2 (Sec. III.B of the paper).

Both models take, for a batch of samples:
  Xo: (B, S, recent_len)  original recent ridership
  Xh: (B, S, hist_len)    historical (same weekday/time, previous week) ridership
  Xt: (B, S, recent_len)  trend component of Xo
  Xr: (B, S, recent_len)  residual component of Xo
and an adjacency matrix (S, S), and predict y_hat: (B, S) -- next-step
ridership for every station simultaneously (many-to-many / network-wide).

DST-TransitNet (V1): 4 independent GRU-then-kGNN branches (one per series:
Xo, Xh, Xt, Xr), sharing one dynamic W_E from the GAT layer, concatenated
into a multi-layer FFNN (Fig. 5).

DST-TransitNetV2: k-GNN spatial aggregation is applied per-timestep (shared
across time) to each series first, the four spatially-aggregated sequences
are concatenated at every timestep, and a single shared GRU -- now placed in
the prediction layer -- extracts the final temporal representation before
the FFNN head (Fig. 6: "the GRU layer ... is moved to the prediction layer,
where it processes the spatially-aggregated features").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import DynamicGATWeights, KGNNLayer, TemporalGRUEncoder


class FFNNHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DSTTransitNet(nn.Module):
    def __init__(self, hist_len: int, mcfg: ModelConfig):
        super().__init__()
        self.gat = DynamicGATWeights(hist_len, mcfg.gat_hidden, mcfg.gat_heads, mcfg.leaky_relu_slope)

        self.gru_o = TemporalGRUEncoder(mcfg.gru_hidden)
        self.gru_h = TemporalGRUEncoder(mcfg.gru_hidden)
        self.gru_t = TemporalGRUEncoder(mcfg.gru_hidden)
        self.gru_r = TemporalGRUEncoder(mcfg.gru_hidden)

        self.gnn_o = KGNNLayer(mcfg.gru_hidden, mcfg.gnn_hidden)
        self.gnn_h = KGNNLayer(mcfg.gru_hidden, mcfg.gnn_hidden)
        self.gnn_t = KGNNLayer(mcfg.gru_hidden, mcfg.gnn_hidden)
        self.gnn_r = KGNNLayer(mcfg.gru_hidden, mcfg.gnn_hidden)

        self.head = FFNNHead(mcfg.gnn_hidden * 4, mcfg.ffnn_hidden, mcfg.ffnn_layers, mcfg.dropout)

    def forward(self, Xo, Xh, Xt, Xr, adjacency):
        W_E = self.gat(Xh, adjacency)  # (B,S,S)

        ho = self.gru_o(Xo.unsqueeze(-1))
        hh = self.gru_h(Xh.unsqueeze(-1))
        ht = self.gru_t(Xt.unsqueeze(-1))
        hr = self.gru_r(Xr.unsqueeze(-1))

        go = self.gnn_o(ho, W_E)
        gh = self.gnn_h(hh, W_E)
        gt = self.gnn_t(ht, W_E)
        gr = self.gnn_r(hr, W_E)

        feat = torch.cat([go, gh, gt, gr], dim=-1)  # (B,S,4*gnn_hidden)
        return self.head(feat)


class DSTTransitNetV2(nn.Module):
    def __init__(self, hist_len: int, mcfg: ModelConfig):
        super().__init__()
        self.gat = DynamicGATWeights(hist_len, mcfg.gat_hidden, mcfg.gat_heads, mcfg.leaky_relu_slope)

        # Per-timestep scalar -> feature projection, shared across series and time.
        proj_dim = mcfg.gnn_hidden
        self.proj_o = nn.Linear(1, proj_dim)
        self.proj_h = nn.Linear(1, proj_dim)
        self.proj_t = nn.Linear(1, proj_dim)
        self.proj_r = nn.Linear(1, proj_dim)

        self.gnn_o = KGNNLayer(proj_dim, proj_dim)
        self.gnn_h = KGNNLayer(proj_dim, proj_dim)
        self.gnn_t = KGNNLayer(proj_dim, proj_dim)
        self.gnn_r = KGNNLayer(proj_dim, proj_dim)

        self.gru = TemporalGRUEncoder(mcfg.gru_hidden, input_dim=proj_dim * 4)
        self.head = FFNNHead(mcfg.gru_hidden, mcfg.ffnn_hidden, mcfg.ffnn_layers, mcfg.dropout)

    def _spatial_seq(self, x, W_E, proj, gnn):
        """x: (B,S,T) -> spatially-aggregated sequence (B,S,T,proj_dim)."""
        B, S, T = x.shape
        z = proj(x.unsqueeze(-1))  # (B,S,T,proj_dim)
        z = z.permute(0, 2, 1, 3).reshape(B * T, S, -1)  # (B*T,S,proj_dim)
        W_E_rep = W_E.unsqueeze(1).expand(B, T, S, S).reshape(B * T, S, S)
        out = gnn(z, W_E_rep)  # (B*T,S,proj_dim)
        out = out.view(B, T, S, -1).permute(0, 2, 1, 3)  # (B,S,T,proj_dim)
        return out

    def forward(self, Xo, Xh, Xt, Xr, adjacency):
        W_E = self.gat(Xh, adjacency)  # (B,S,S), computed once, applied at every timestep

        so = self._spatial_seq(Xo, W_E, self.proj_o, self.gnn_o)
        sh = self._spatial_seq(Xh, W_E, self.proj_h, self.gnn_h)
        st = self._spatial_seq(Xt, W_E, self.proj_t, self.gnn_t)
        sr = self._spatial_seq(Xr, W_E, self.proj_r, self.gnn_r)

        # Align time lengths (Xh may have a different length than Xo/Xt/Xr);
        # truncate/pad to the shortest common length T_min via right-alignment
        # (most recent T_min steps of each series).
        T_min = min(so.shape[2], sh.shape[2], st.shape[2], sr.shape[2])
        seq = torch.cat([
            so[:, :, -T_min:, :], sh[:, :, -T_min:, :],
            st[:, :, -T_min:, :], sr[:, :, -T_min:, :],
        ], dim=-1)  # (B,S,T_min,4*proj_dim)

        out_seq = self.gru.forward_sequence(seq)  # (B,S,T_min,gru_hidden)
        h_last = out_seq[:, :, -1, :]  # (B,S,gru_hidden)
        return self.head(h_last)
