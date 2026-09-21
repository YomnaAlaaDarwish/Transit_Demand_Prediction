"""Core building blocks of DST-TransitNet: GAT-based dynamic spatial weight
calculation and the k-GNN spatial aggregation operator, following the
equations in Sec. III.A/III.B of Wang & Shalaby (2024, arXiv:2410.15013).

GRU is used directly via torch.nn.GRU (its equations, Sec. III.A.1, match
the standard PyTorch GRU cell exactly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGATWeights(nn.Module):
    """Dynamic Spatial Weight Calculation layer (Sec. III.B.2).

    Computes attention-based edge weights W_E over the (static) BRT graph
    from historical passenger counts X_h, following the GAT formulation in
    Sec. III.A.3:
        e_ij = LeakyReLU(a^T [z_i || z_j]),         z_i = W h_i
        W_E_ij = softmax_j(e_ij)  over j in N(i)
    Multi-head attention is used for extra capacity (standard GAT practice);
    head outputs are averaged into a single W_E used by all four
    spatio-temporal aggregation branches, matching the paper's description
    of a single shared W_E feeding every k-GNN branch.
    """

    def __init__(self, hist_len: int, hidden_dim: int, n_heads: int = 4, leaky_slope: float = 0.01):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hist_len, hidden_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, hidden_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, hidden_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_slope = leaky_slope

    def forward(self, Xh: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Xh: (B, S, Nh) historical ridership features per station.
        adjacency: (S, S) binary adjacency (with self-loops); 1 = connected.
        Returns W_E: (B, S, S) attention weights, zero outside the graph.
        """
        B, S, _ = Xh.shape
        z = self.W(Xh).view(B, S, self.n_heads, self.hidden_dim)  # (B,S,H,D)

        e_src = torch.einsum("bshd,hd->bsh", z, self.a_src)  # (B,S,H)
        e_dst = torch.einsum("bshd,hd->bsh", z, self.a_dst)  # (B,S,H)
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (B,S_i,S_j,H)
        e = F.leaky_relu(e, negative_slope=self.leaky_slope)

        mask = (adjacency > 0).unsqueeze(0).unsqueeze(-1)  # (1,S,S,1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)  # softmax over neighbors j, per head
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated nodes -> all -inf row
        W_E = alpha.mean(dim=-1)  # average heads -> (B, S_i, S_j)
        return W_E


class KGNNLayer(nn.Module):
    """k-dimensional GNN spatial aggregation (Sec. III.A.2 / III.B.3):
        h_i' = sigma( x_i W1 + sum_{j in N(i)} W_E[j,i] * x_j  W2 )
    Uses the dynamic, sample-specific W_E produced by DynamicGATWeights
    (falls back to the plain normalized adjacency if W_E is None, used by
    baselines/ablations).
    """

    def __init__(self, in_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=True)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, F)
        edge_weight: (B, S, S) or (S, S); edge_weight[b, i, j] weights the
            contribution of source node j to target node i (row i sums to 1
            over j, matching DynamicGATWeights' softmax-over-neighbors
            convention: W_E[b, i, :] is node i's attention distribution
            over its neighbors).
        """
        if edge_weight.dim() == 2:
            edge_weight = edge_weight.unsqueeze(0)
        # aggregate neighbor features: out[b,i,:] = sum_j edge_weight[b,i,j] * x[b,j,:]
        neigh = torch.einsum("bij,bjf->bif", edge_weight, x)
        out = self.W1(x) + self.W2(neigh)
        if self.activation is not None:
            out = self.activation(out)
        return out


class TemporalGRUEncoder(nn.Module):
    """Applies a shared-weight GRU independently to every station's scalar
    time series (Sec. III.A.1). Input (B, S, T, 1) -> output (B, S, hidden).
    """

    def __init__(self, hidden_dim: int, input_dim: int = 1, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, T, F_ = x.shape
        x = x.reshape(B * S, T, F_)
        _, h_n = self.gru(x)
        h = h_n[-1].view(B, S, -1)
        return h

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the full output sequence instead of only the last hidden
        state -- used by DST-TransitNetV2's prediction-layer GRU which
        consumes a per-timestep spatially-aggregated sequence.
        x: (B, S, T, F) or (B, T, F) -> (B, [S,] T, hidden)
        """
        if x.dim() == 4:
            B, S, T, F_ = x.shape
            x = x.reshape(B * S, T, F_)
            out, _ = self.gru(x)
            return out.view(B, S, T, -1)
        else:
            out, _ = self.gru(x)
            return out


def moving_average_decompose_torch(x: torch.Tensor, kernel: int):
    """Torch counterpart of dst_transitnet.data.moving_average_decompose, for
    use inside the long-term iterative-forecasting loop where recent inputs
    are updated with the model's own previous predictions on the fly.
    x: (..., T)
    """
    pad_left = kernel // 2
    pad_right = kernel - 1 - pad_left
    x_p = F.pad(x, (pad_left, pad_right), mode="replicate")
    trend = F.avg_pool1d(x_p.reshape(-1, 1, x_p.shape[-1]), kernel_size=kernel, stride=1)
    trend = trend.reshape(*x.shape)
    residual = x - trend
    return trend, residual
