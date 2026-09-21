"""Training loop, evaluation, and the long-term iterative forecasting
framework (Sec. III.B.4) for DST-TransitNet / baselines."""
import json
import logging
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .data import WindowedSamples
from .layers import moving_average_decompose_torch
from .metrics import r2_score, maape

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _batches(n, batch_size, shuffle=False, seed=0):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def to_tensor(w: WindowedSamples, indices=None):
    if indices is None:
        indices = slice(None)
    Xo = torch.from_numpy(w.Xo[indices]).transpose(1, 2)  # (N, S, T)
    Xh = torch.from_numpy(w.Xh[indices]).transpose(1, 2)
    Xt = torch.from_numpy(w.Xt[indices]).transpose(1, 2)
    Xr = torch.from_numpy(w.Xr[indices]).transpose(1, 2)
    y = torch.from_numpy(w.y[indices])
    return Xo, Xh, Xt, Xr, y


def train_model(model: nn.Module, train_w: WindowedSamples, val_w: WindowedSamples,
                 adjacency: np.ndarray, tcfg: TrainConfig, log_path: str = None,
                 device: str = "cpu"):
    set_seed(tcfg.seed)
    model = model.to(device)
    A = torch.from_numpy(adjacency).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loss_fn = nn.MSELoss()

    Xo_tr, Xh_tr, Xt_tr, Xr_tr, y_tr = to_tensor(train_w)
    Xo_va, Xh_va, Xt_va, Xr_va, y_va = to_tensor(val_w)

    best_val = float("inf")
    best_state = None
    patience_left = tcfg.patience
    history = []

    n = Xo_tr.shape[0]
    for epoch in range(tcfg.max_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for idx in _batches(n, tcfg.batch_size, shuffle=True, seed=tcfg.seed + epoch):
            opt.zero_grad()
            xo, xh, xt, xr = Xo_tr[idx].to(device), Xh_tr[idx].to(device), Xt_tr[idx].to(device), Xr_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            pred = model(xo, xh, xt, xr, A)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            epoch_loss += loss.item() * len(idx)
            n_batches += 1
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_preds = []
            for idx in _batches(Xo_va.shape[0], tcfg.batch_size):
                xo, xh, xt, xr = Xo_va[idx].to(device), Xh_va[idx].to(device), Xt_va[idx].to(device), Xr_va[idx].to(device)
                val_preds.append(model(xo, xh, xt, xr, A).cpu())
            val_pred = torch.cat(val_preds).numpy()
            val_loss = float(np.mean((val_pred - y_va.numpy()) ** 2))

        dt = time.time() - t0
        logger.info("epoch %d train_mse=%.6f val_mse=%.6f (%.1fs)", epoch, epoch_loss, val_loss, dt)
        history.append({"epoch": epoch, "train_mse": epoch_loss, "val_mse": val_loss, "seconds": dt})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = tcfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if log_path:
        with open(log_path, "w") as f:
            json.dump({"history": history, "config": asdict(tcfg)}, f, indent=2)

    return model, history


@torch.no_grad()
def predict(model: nn.Module, w: WindowedSamples, adjacency: np.ndarray,
            batch_size: int = 128, device: str = "cpu") -> np.ndarray:
    model.eval()
    A = torch.from_numpy(adjacency).to(device)
    Xo, Xh, Xt, Xr, _ = to_tensor(w)
    preds = []
    for idx in _batches(Xo.shape[0], batch_size):
        xo, xh, xt, xr = Xo[idx].to(device), Xh[idx].to(device), Xt[idx].to(device), Xr[idx].to(device)
        preds.append(model(xo, xh, xt, xr, A).cpu())
    return torch.cat(preds).numpy()


def evaluate(model: nn.Module, w: WindowedSamples, adjacency: np.ndarray, device: str = "cpu"):
    pred = predict(model, w, adjacency, device=device)
    return {
        "r2": r2_score(w.y, pred),
        "maape": maape(w.y, pred),
    }, pred


@torch.no_grad()
def long_term_forecast(model: nn.Module, w: WindowedSamples, adjacency: np.ndarray,
                        decomposition_kernel: int, max_lag: int, hist_len: int, device: str = "cpu",
                        batch_size: int = 64) -> np.ndarray:
    """Iterative long-term forecasting (Sec. III.B.4 / Fig. 7): recent input
    Xo is rolled forward using the model's own 1-step predictions; Xh is
    always the *true* previous-week observation for each future target step
    (already available -- never a model prediction), matching the paper's
    deployment design (Sec. III.C).

    w must be built with `long_term=True`: w.y has shape (N, max_lag, S) and
    w.Xh has shape (N, hist_len + max_lag - 1, S) so that lag L's true
    previous-week window is the slice w.Xh[:, L:L+hist_len, :].

    Returns predictions of shape (N, max_lag, S).
    """
    model.eval()
    A = torch.from_numpy(adjacency).to(device)
    Xo, Xh_full, _, _, _ = to_tensor(w)  # Xh_full: (N,S,hist_len+max_lag-1)
    N, S, recent_len = Xo.shape

    all_preds = np.zeros((N, max_lag, S), dtype=np.float32)

    for idx in _batches(N, batch_size):
        xo = Xo[idx].clone().to(device)  # (b,S,recent_len), rolled forward each lag
        xh_full_b = Xh_full[idx].to(device)  # (b,S,hist_len+max_lag-1)
        for lag in range(max_lag):
            xh = xh_full_b[:, :, lag: lag + hist_len]  # true previous-week window for this lag
            xt, xr = moving_average_decompose_torch(xo, decomposition_kernel)
            pred = model(xo, xh, xt, xr, A)  # (b,S)
            all_preds[idx, lag, :] = pred.cpu().numpy()
            xo = torch.cat([xo[:, :, 1:], pred.unsqueeze(-1)], dim=-1)

    return all_preds
