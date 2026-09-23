"""Time2Vec-inspired learned temporal embedding (Sec. III.B.1 of the paper).

The paper's description (quoted, Sec. III.B.1, steps 1-6) is a *pointwise*
feedforward encoder, not a recurrent/attention model: "two linear layers
with ReLU activation" applied over "a subset of temporally relevant numeric
features" collected via "a 48-step window", trained with "an MSE loss
between sequential embeddings to ensure temporal continuity", after which
"the final 64-dimensional output corresponding to the latest time step in
each sequence is retained for forecasting."

Read literally, the encoder itself has no mechanism to mix information
*across* window positions (no recurrence, no attention, just two Linear+ReLU
layers) -- the sliding window is the *training* signal (many (t, t+1) pairs
to train a pointwise next-step predictor on), and at inference only the
current timestep's raw feature vector is actually encoded. This is exactly
what we implement:

  encoder(x_t) = Linear(hidden, embed_dim)(ReLU(Linear(in_dim, hidden)(x_t)))

trained via a linear decoder head, `decoder(encoder(x_t)) ~= x_{t+1}`
(MSE loss), which is our concrete, documented reconstruction of "MSE loss
between sequential embeddings to ensure temporal continuity" -- the encoder
must retain enough information to predict what comes next.

Trained ONCE, globally, pooling (t, t+1) pairs from every station's training
period (not per-station) -- a shared temporal-pattern feature extractor,
analogous to the weight-sharing compute-budget adaptation already documented
for the DST-TransitNet reproduction's baselines. The station-specific
prediction happens downstream, in the per-station stacked ensemble
(ensemble.py), which is trained independently per station.
"""
import numpy as np
import torch
import torch.nn as nn


class Time2VecEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.decoder = nn.Linear(embed_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def embed(self, x):
        with torch.no_grad():
            return self.encoder(x)


def build_pointwise_features(values: np.ndarray, values_week_ago: np.ndarray) -> np.ndarray:
    """Builds the per-timestep raw feature vector fed to the Time2Vec
    encoder: [current scaled ridership, same-time-last-week scaled
    ridership]. These are the two "temporally relevant numeric features"
    available without weather/news (see config.py docstring) -- the
    analogue of the paper's ['actual_load_MW', 'actual_load_lag_1week', ...]
    selection, restricted to what this dataset actually offers.
    """
    return np.stack([values, values_week_ago], axis=-1).astype(np.float32)


def train_time2vec(train_pairs_x: np.ndarray, train_pairs_y: np.ndarray, cfg,
                    device: str = "cpu") -> Time2VecEncoder:
    """train_pairs_x: (N, in_dim) raw feature vectors at t.
    train_pairs_y: (N, in_dim) raw feature vectors at t+1 (next-step target).
    """
    model = Time2VecEncoder(train_pairs_x.shape[1], cfg.time2vec_hidden, cfg.time2vec_embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.time2vec_lr)
    loss_fn = nn.MSELoss()

    x = torch.from_numpy(train_pairs_x)
    y = torch.from_numpy(train_pairs_y)
    n = x.shape[0]

    rng = np.random.default_rng(0)
    for epoch in range(cfg.time2vec_epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, cfg.time2vec_batch_size):
            batch_idx = idx[i:i + cfg.time2vec_batch_size]
            xb, yb = x[batch_idx].to(device), y[batch_idx].to(device)
            opt.zero_grad()
            _, x_hat = model(xb)
            loss = loss_fn(x_hat, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= n
        print(f"[time2vec] epoch {epoch} mse={epoch_loss:.6f}")

    return model


@torch.no_grad()
def embed_features(model: Time2VecEncoder, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    t = torch.from_numpy(x.astype(np.float32)).to(device)
    z = model.embed(t)
    return z.cpu().numpy()
