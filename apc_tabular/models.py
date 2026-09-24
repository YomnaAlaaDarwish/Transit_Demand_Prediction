"""The paper's five algorithms (Sec. 2.1): Tabular DNN, CatBoost, Random
Forest, XGBoost, LightGBM. Each wrapper exposes a common `.fit(df, y)` /
`.predict(df)` interface operating directly on the long-format dataframe
from features.py (categorical columns kept as pandas `category` dtype).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from .config import ModelConfig
from .features import CATEGORICAL_COLS, CONTINUOUS_COLS, BOOLEAN_COLS, ALL_FEATURE_COLS


class CatBoostWrapper:
    def __init__(self, mcfg: ModelConfig):
        self.model = CatBoostRegressor(
            iterations=mcfg.catboost_iterations, depth=mcfg.catboost_depth,
            l2_leaf_reg=mcfg.catboost_l2_leaf_reg, learning_rate=mcfg.catboost_learning_rate,
            bagging_temperature=mcfg.catboost_bagging_temperature,
            cat_features=CATEGORICAL_COLS, loss_function="RMSE",
            random_seed=mcfg.random_state, verbose=False, thread_count=-1)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Xc = X.copy()
        for c in CATEGORICAL_COLS:
            Xc[c] = Xc[c].astype(str)
        self.model.fit(Xc[ALL_FEATURE_COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X.copy()
        for c in CATEGORICAL_COLS:
            Xc[c] = Xc[c].astype(str)
        return self.model.predict(Xc[ALL_FEATURE_COLS])


class LightGBMWrapper:
    def __init__(self, mcfg: ModelConfig):
        self.model = LGBMRegressor(
            n_estimators=mcfg.lgbm_n_estimators, max_depth=mcfg.lgbm_max_depth,
            num_leaves=mcfg.lgbm_num_leaves, learning_rate=mcfg.lgbm_learning_rate,
            min_child_samples=mcfg.lgbm_min_child_samples, reg_alpha=mcfg.lgbm_reg_alpha,
            reg_lambda=mcfg.lgbm_reg_lambda, subsample=mcfg.lgbm_subsample,
            colsample_bytree=mcfg.lgbm_colsample_bytree, random_state=mcfg.random_state,
            n_jobs=-1, verbosity=-1)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X[ALL_FEATURE_COLS], y, categorical_feature=CATEGORICAL_COLS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[ALL_FEATURE_COLS])


class XGBoostWrapper:
    def __init__(self, mcfg: ModelConfig):
        self.model = XGBRegressor(
            n_estimators=mcfg.xgb_n_estimators, max_depth=mcfg.xgb_max_depth,
            colsample_bytree=mcfg.xgb_colsample_bytree, learning_rate=mcfg.xgb_eta,
            gamma=mcfg.xgb_gamma, min_child_weight=mcfg.xgb_min_child_weight,
            reg_alpha=mcfg.xgb_reg_alpha, reg_lambda=mcfg.xgb_reg_lambda,
            subsample=mcfg.xgb_subsample, random_state=mcfg.random_state,
            n_jobs=-1, tree_method="hist", enable_categorical=True, verbosity=0)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X[ALL_FEATURE_COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[ALL_FEATURE_COLS])


class RandomForestWrapper:
    """RF has no native categorical support in scikit-learn -- ordinal-encode
    (fit on training data only, unseen categories at eval time map to -1)."""

    def __init__(self, mcfg: ModelConfig):
        self.model = RandomForestRegressor(
            n_estimators=mcfg.rf_n_estimators, max_depth=mcfg.rf_max_depth,
            max_features=mcfg.rf_max_features, max_samples=mcfg.rf_max_samples,
            min_samples_leaf=mcfg.rf_min_samples_leaf, min_samples_split=mcfg.rf_min_samples_split,
            random_state=mcfg.random_state, n_jobs=-1, bootstrap=True)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        cat_enc = self.encoder.fit_transform(X[CATEGORICAL_COLS])
        Xnum = np.hstack([cat_enc, X[CONTINUOUS_COLS + BOOLEAN_COLS].values])
        self.model.fit(Xnum, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cat_enc = self.encoder.transform(X[CATEGORICAL_COLS])
        Xnum = np.hstack([cat_enc, X[CONTINUOUS_COLS + BOOLEAN_COLS].values])
        return self.model.predict(Xnum)


class TabularDNN(nn.Module):
    """Sec. 2.1.1 / Fig. 1: embedding layer per categorical feature,
    concatenated with continuous features, passed through `n_blocks` of
    Linear+ReLU+BatchNorm+Dropout, single regression output (Boarding only
    -- see config.py for why Alighting is out of scope here).
    """

    def __init__(self, cat_cardinalities: dict, n_continuous: int, mcfg: ModelConfig):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(card + 1, min(mcfg.dnn_embedding_dim, (card + 1) // 2 + 1))
            for c, card in cat_cardinalities.items()
        })
        embed_dim_total = sum(e.embedding_dim for e in self.embeddings.values())
        in_dim = embed_dim_total + n_continuous

        blocks = []
        d = in_dim
        for h in mcfg.dnn_hidden_sizes:
            blocks += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(mcfg.dnn_dropout)]
            d = h
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d, 1)

    def forward(self, cat_inputs: dict, cont_input: torch.Tensor):
        embs = [self.embeddings[c](cat_inputs[c]) for c in self.embeddings]
        z = torch.cat(embs + [cont_input], dim=1)
        z = self.blocks(z)
        return torch.relu(self.head(z)).squeeze(-1)  # counts are non-negative


class TabularDNNWrapper:
    def __init__(self, mcfg: ModelConfig, device: str = "cpu"):
        self.mcfg = mcfg
        self.device = device
        self.encoders = {c: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                          for c in CATEGORICAL_COLS}
        self.scaler = StandardScaler()
        self.model = None

    def _prep(self, X: pd.DataFrame, fit: bool):
        cat_arrays = {}
        for c in CATEGORICAL_COLS:
            col = X[[c]]
            arr = self.encoders[c].fit_transform(col) if fit else self.encoders[c].transform(col)
            cat_arrays[c] = (arr.flatten() + 1).astype(np.int64)  # +1: reserve 0 for unseen(-1)->0
        cont = X[CONTINUOUS_COLS + BOOLEAN_COLS].values.astype(np.float32)
        cont = self.scaler.fit_transform(cont) if fit else self.scaler.transform(cont)
        return cat_arrays, cont.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None):
        cat_arrays, cont = self._prep(X, fit=True)
        cardinalities = {c: int(cat_arrays[c].max()) for c in CATEGORICAL_COLS}
        self.model = TabularDNN(cardinalities, cont.shape[1], self.mcfg).to(self.device)

        y_mean, y_std = y.mean(), y.std() + 1e-6
        self.y_mean, self.y_std = y_mean, y_std
        y_norm = ((y - y_mean) / y_std).astype(np.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.mcfg.dnn_lr)
        loss_fn = nn.MSELoss()
        n = len(y)
        bs = self.mcfg.dnn_batch_size

        cont_t = torch.from_numpy(cont)
        cat_t = {c: torch.from_numpy(v) for c, v in cat_arrays.items()}
        y_t = torch.from_numpy(y_norm)

        if X_val is not None:
            cat_val, cont_val = self._prep(X_val, fit=False)
            cont_val_t = torch.from_numpy(cont_val).to(self.device)
            cat_val_t = {c: torch.from_numpy(v).to(self.device) for c, v in cat_val.items()}
            y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)

        best_val, patience_left, best_state = float("inf"), self.mcfg.dnn_patience, None
        rng = np.random.default_rng(self.mcfg.random_state)

        for epoch in range(self.mcfg.dnn_max_epochs):
            self.model.train()
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, bs):
                b = idx[i:i + bs]
                opt.zero_grad()
                cat_b = {c: cat_t[c][b].to(self.device) for c in cat_t}
                pred = self.model(cat_b, cont_t[b].to(self.device))
                loss = loss_fn(pred, y_t[b].to(self.device))
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(b)
            epoch_loss /= n

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(cat_val_t, cont_val_t).cpu().numpy()
                val_mse = float(np.mean((val_pred - y_val_norm) ** 2))
                print(f"[tabular_dnn] epoch {epoch} train_mse={epoch_loss:.4f} val_mse={val_mse:.4f}")
                if val_mse < best_val - 1e-5:
                    best_val = val_mse
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.mcfg.dnn_patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
            else:
                print(f"[tabular_dnn] epoch {epoch} train_mse={epoch_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cat_arrays, cont = self._prep(X, fit=False)
        self.model.eval()
        with torch.no_grad():
            cat_t = {c: torch.from_numpy(v).to(self.device) for c, v in cat_arrays.items()}
            cont_t = torch.from_numpy(cont).to(self.device)
            pred = self.model(cat_t, cont_t).cpu().numpy()
        return pred * self.y_std + self.y_mean


def build_model(name: str, mcfg: ModelConfig):
    if name == "catboost":
        return CatBoostWrapper(mcfg)
    if name == "lightgbm":
        return LightGBMWrapper(mcfg)
    if name == "xgboost":
        return XGBoostWrapper(mcfg)
    if name == "random_forest":
        return RandomForestWrapper(mcfg)
    if name == "tabular_dnn":
        return TabularDNNWrapper(mcfg)
    raise ValueError(name)
