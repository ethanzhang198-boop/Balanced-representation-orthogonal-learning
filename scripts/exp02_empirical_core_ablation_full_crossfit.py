from __future__ import annotations

"""Representation-specification stability experiment."""

import argparse
import json
import math
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data_utils import prepare_dataset, set_seed
from src.output_utils import write_workbook


SCRIPT_VERSION = "CORE_SPEC_STABILITY_FULL_CROSSFIT_20260723"
OUT_DIRNAME = "exp02_empirical_core_ablation_full_crossfit"

SPEC_LIBRARY: dict[str, dict[str, str]] = {
    "plain_ft": {
        "spec_id": "plain_ft",
        "name": "Fully cross-fitted Plain FT-DML",
        "backbone": "ft",
        "balance": "none",
    },
    "mlp_sinkhorn": {
        "spec_id": "mlp_sinkhorn",
        "name": "Fully cross-fitted BROL-MLP-Sinkhorn",
        "backbone": "mlp",
        "balance": "sinkhorn",
    },
    "ft_mmd": {
        "spec_id": "ft_mmd",
        "name": "Fully cross-fitted BROL-FT-MMD",
        "backbone": "ft",
        "balance": "mmd",
    },
    "ft_sinkhorn": {
        "spec_id": "ft_sinkhorn",
        "name": "Fully cross-fitted BROL-FT-Sinkhorn",
        "backbone": "ft",
        "balance": "sinkhorn",
    },
}


def summarize_effect(model_name: str, result: dict[str, Any], spec_id: str) -> dict[str, Any]:
    estimate = float(result["estimate"])
    ci_low = float(result["ci_low"])
    ci_high = float(result["ci_high"])
    return {
        "spec_id": spec_id,
        "model": model_name,
        "estimate": estimate,
        "std_err": float(result["std_err"]),
        "t_value": float(result["t_value"]),
        "p_value": float(result["p_value"]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "nobs": int(result["nobs"]),
        "sign": "positive" if estimate > 0 else ("negative" if estimate < 0 else "zero"),
        "significant_5pct": bool(float(result["p_value"]) < 0.05),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def ols_ate(X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([T.reshape(-1, 1), X])
    columns = ["treatment"] + [f"x_{i}" for i in range(X.shape[1])]
    fit = sm.OLS(
        Y,
        sm.add_constant(pd.DataFrame(design, columns=columns), has_constant="add"),
    ).fit(cov_type="HC3")
    ci = fit.conf_int()
    return {
        "estimate": float(fit.params["treatment"]),
        "std_err": float(fit.bse["treatment"]),
        "t_value": float(fit.tvalues["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
        "ci_low": float(ci.loc["treatment", 0]),
        "ci_high": float(ci.loc["treatment", 1]),
        "const": float(fit.params["const"]),
        "nobs": int(fit.nobs),
        "rsquared": float(fit.rsquared),
        "rsquared_adj": float(fit.rsquared_adj),
    }


def residual_hc3(
    Y: np.ndarray,
    T: np.ndarray,
    y_hat: np.ndarray,
    t_hat: np.ndarray,
) -> dict[str, Any]:
    y_res = np.asarray(Y, dtype=np.float64) - np.asarray(y_hat, dtype=np.float64)
    t_res = np.asarray(T, dtype=np.float64) - np.asarray(t_hat, dtype=np.float64)
    fit = sm.OLS(
        y_res,
        sm.add_constant(pd.DataFrame({"t_res": t_res}), has_constant="add"),
    ).fit(cov_type="HC3")
    ci = fit.conf_int()
    return {
        "estimate": float(fit.params["t_res"]),
        "std_err": float(fit.bse["t_res"]),
        "t_value": float(fit.tvalues["t_res"]),
        "p_value": float(fit.pvalues["t_res"]),
        "ci_low": float(ci.loc["t_res", 0]),
        "ci_high": float(ci.loc["t_res", 1]),
        "const": float(fit.params["const"]),
        "nobs": int(fit.nobs),
        "y_res_mean": float(np.mean(y_res)),
        "t_res_mean": float(np.mean(t_res)),
        "t_res_variance": float(np.var(t_res, ddof=0)),
    }


def dml_ate_lasso_quiet(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    folds: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    y_hat = np.zeros(len(Y), dtype=np.float64)
    t_hat = np.zeros(len(T), dtype=np.float64)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (tr, te) in enumerate(kf.split(X)):
            ym = LassoCV(cv=3, random_state=seed + fold, max_iter=10000, n_jobs=1)
            tm = LassoCV(cv=3, random_state=seed + fold + 100, max_iter=10000, n_jobs=1)
            ym.fit(X[tr], Y[tr])
            tm.fit(X[tr], T[tr])
            y_hat[te] = ym.predict(X[te])
            t_hat[te] = tm.predict(X[te])
    return residual_hc3(Y, T, y_hat, t_hat)


def standardize_from_train(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = float(np.mean(train))
    sd = float(np.std(train, ddof=0))
    if sd < 1e-8:
        sd = 1.0
    return tuple((np.asarray(arr) - mean) / sd for arr in arrays)


def fit_quantile_cutoffs(values: np.ndarray, n_groups: int) -> np.ndarray:
    probabilities = np.arange(1, n_groups, dtype=np.float64) / n_groups
    return np.quantile(np.asarray(values, dtype=np.float64), probabilities)


def apply_quantile_groups(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return np.digitize(
        np.asarray(values, dtype=np.float64),
        np.asarray(cutoffs, dtype=np.float64),
        right=True,
    ).astype(np.int64)


def sinkhorn_pair_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)
    cost = torch.cdist(x, y, p=2).pow(2)
    a = torch.full((x.size(0),), 1.0 / x.size(0), device=x.device)
    b = torch.full((y.size(0),), 1.0 / y.size(0), device=y.device)
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (kernel @ v + 1e-8)
        v = b / (kernel.t() @ u + 1e-8)
    return torch.sum(u[:, None] * kernel * v[None, :] * cost)


def sinkhorn_quantile_balance_loss(
    z: torch.Tensor,
    group: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    present = torch.unique(group)
    if present.numel() < 2:
        return z.new_tensor(0.0)
    z_for_balance = F.normalize(z, p=2, dim=1)
    losses: list[torch.Tensor] = []
    for i in range(present.numel()):
        for j in range(i + 1, present.numel()):
            zi = z_for_balance[group == present[i]]
            zj = z_for_balance[group == present[j]]
            losses.append(
                sinkhorn_pair_loss(
                    zi,
                    zj,
                    epsilon=epsilon,
                    n_iters=n_iters,
                )
            )
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


def mmd_pair_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...],
) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    loss = x.new_tensor(0.0)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        loss = loss + (
            torch.exp(-gamma * xx).mean()
            + torch.exp(-gamma * yy).mean()
            - 2.0 * torch.exp(-gamma * xy).mean()
        )
    return loss / len(sigmas)


def mmd_quantile_balance_loss(
    z: torch.Tensor,
    group: torch.Tensor,
    sigmas: tuple[float, ...],
) -> torch.Tensor:
    present = torch.unique(group)
    if present.numel() < 2:
        return z.new_tensor(0.0)
    z_for_balance = F.normalize(z, p=2, dim=1)
    losses: list[torch.Tensor] = []
    for i in range(present.numel()):
        for j in range(i + 1, present.numel()):
            zi = z_for_balance[group == present[i]]
            zj = z_for_balance[group == present[j]]
            losses.append(mmd_pair_loss(zi, zj, sigmas=sigmas))
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


def compute_balance_loss(
    z: torch.Tensor,
    group: torch.Tensor,
    balance_type: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    if balance_type == "none":
        return z.new_tensor(0.0)
    if balance_type == "sinkhorn":
        return sinkhorn_quantile_balance_loss(
            z,
            group,
            epsilon=args.sinkhorn_epsilon,
            n_iters=args.sinkhorn_iters,
        )
    if balance_type == "mmd":
        return mmd_quantile_balance_loss(
            z,
            group,
            sigmas=tuple(float(value) for value in args.mmd_sigmas),
        )
    raise ValueError(f"Unknown balance_type: {balance_type}")


def balance_weight_for_spec(spec: dict[str, str], args: argparse.Namespace) -> float:
    if spec["balance"] == "none":
        return 0.0
    if spec["balance"] == "sinkhorn":
        return float(args.sinkhorn_weight)
    if spec["balance"] == "mmd":
        return float(args.mmd_weight)
    raise ValueError(f"Unknown balance_type: {spec['balance']}")


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FeatureTransformerNBR(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
        )
        self.treatment_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(latent_dim + 1),
            nn.Linear(latent_dim + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_vec = encoded[:, 0, :]
        feature_tokens = encoded[:, 1:, :]
        weights = torch.softmax(self.attn_pool(feature_tokens).squeeze(-1), dim=1)
        attentive = torch.sum(feature_tokens * weights.unsqueeze(-1), dim=1)
        return self.projection(torch.cat([cls_vec, attentive], dim=1))

    def forward(
        self,
        x: torch.Tensor,
        treatment_for_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        d_hat = self.treatment_head(z)
        y_hat = self.outcome_head(torch.cat([z, treatment_for_head], dim=1))
        return z, d_hat, y_hat


class MLPBalancingNBR(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.treatment_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(latent_dim + 1),
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        treatment_for_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        d_hat = self.treatment_head(z)
        y_hat = self.outcome_head(torch.cat([z, treatment_for_head], dim=1))
        return z, d_hat, y_hat


def make_model(
    spec: dict[str, str],
    n_features: int,
    args: argparse.Namespace,
) -> nn.Module:
    if spec["backbone"] == "ft":
        return FeatureTransformerNBR(
            n_features=n_features,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )
    if spec["backbone"] == "mlp":
        return MLPBalancingNBR(
            n_features=n_features,
            hidden_dim=args.mlp_hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown backbone: {spec['backbone']}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        device_arg = "cuda:0"
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


def encode_model(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.encode(x).detach().cpu().numpy())
    return np.vstack(outputs)


def fit_encoder_on_outer_training_fold(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    T_train: np.ndarray,
    spec: dict[str, str],
    args: argparse.Namespace,
    seed: int,
    outer_fold: int,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Fit and tune one encoder using only the current outer-training fold."""
    all_idx = np.arange(len(X_train))
    fit_idx, valid_idx = train_test_split(
        all_idx,
        test_size=args.inner_valid_fraction,
        shuffle=True,
        random_state=seed,
    )

    x_fit = np.asarray(X_train[fit_idx], dtype=np.float32)
    x_valid = np.asarray(X_train[valid_idx], dtype=np.float32)
    y_fit = np.asarray(Y_train[fit_idx], dtype=np.float64)
    y_valid = np.asarray(Y_train[valid_idx], dtype=np.float64)
    t_fit = np.asarray(T_train[fit_idx], dtype=np.float64)
    t_valid = np.asarray(T_train[valid_idx], dtype=np.float64)

    y_fit_s, y_valid_s = standardize_from_train(y_fit, y_fit, y_valid)
    t_fit_s, t_valid_s = standardize_from_train(t_fit, t_fit, t_valid)
    cutoffs = fit_quantile_cutoffs(t_fit, args.treatment_groups)
    fit_groups = apply_quantile_groups(t_fit, cutoffs)
    valid_groups = apply_quantile_groups(t_valid, cutoffs)

    device = resolve_device(args.device)
    set_seed(seed)
    model = make_model(spec, X_train.shape[1], args).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    xtr = torch.tensor(x_fit, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_fit_s.reshape(-1, 1), dtype=torch.float32, device=device)
    ttr = torch.tensor(t_fit_s.reshape(-1, 1), dtype=torch.float32, device=device)
    gtr = torch.tensor(fit_groups, dtype=torch.long, device=device)
    xva = torch.tensor(x_valid, dtype=torch.float32, device=device)
    yva = torch.tensor(y_valid_s.reshape(-1, 1), dtype=torch.float32, device=device)
    tva = torch.tensor(t_valid_s.reshape(-1, 1), dtype=torch.float32, device=device)
    gva = torch.tensor(valid_groups, dtype=torch.long, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []
    n_batches = math.ceil(xtr.size(0) / args.batch_size)
    active_balance_weight = balance_weight_for_spec(spec, args)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(xtr.size(0), device=device)
        totals = {
            "loss": 0.0,
            "outcome": 0.0,
            "treatment": 0.0,
            "balance": 0.0,
            "stability": 0.0,
        }

        for batch_no in range(n_batches):
            idx = order[
                batch_no * args.batch_size : (batch_no + 1) * args.batch_size
            ]
            bx, by, bt, bg = xtr[idx], ytr[idx], ttr[idx], gtr[idx]
            optimizer.zero_grad(set_to_none=True)
            z, d_hat, y_hat = model(bx, bt)
            outcome_loss = F.smooth_l1_loss(y_hat, by)
            treatment_loss = F.smooth_l1_loss(d_hat, bt)
            bal_loss = compute_balance_loss(z, bg, spec["balance"], args)

            if args.stability_weight > 0:
                noise = torch.randn_like(bx) * args.perturb_std
                z_pert = model.encode(bx + noise)
                stability_loss = F.mse_loss(z_pert, z.detach())
            else:
                stability_loss = z.new_tensor(0.0)

            loss = (
                args.outcome_weight * outcome_loss
                + args.treatment_weight * treatment_loss
                + active_balance_weight * bal_loss
                + args.stability_weight * stability_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            n_batch = bx.size(0)
            totals["loss"] += float(loss.item()) * n_batch
            totals["outcome"] += float(outcome_loss.item()) * n_batch
            totals["treatment"] += float(treatment_loss.item()) * n_batch
            totals["balance"] += float(bal_loss.item()) * n_batch
            totals["stability"] += float(stability_loss.item()) * n_batch

        model.eval()
        with torch.no_grad():
            zva, dva_hat, yva_hat = model(xva, tva)
            valid_outcome = mean_squared_error(
                yva.detach().cpu().numpy().reshape(-1),
                yva_hat.detach().cpu().numpy().reshape(-1),
            )
            valid_treatment = mean_squared_error(
                tva.detach().cpu().numpy().reshape(-1),
                dva_hat.detach().cpu().numpy().reshape(-1),
            )
            valid_balance = float(
                compute_balance_loss(zva, gva, spec["balance"], args).item()
            )

        valid_score = (
            args.outcome_weight * valid_outcome
            + args.treatment_weight * valid_treatment
            + active_balance_weight * valid_balance
        )
        row = {
            "spec_id": spec["spec_id"],
            "model": spec["name"],
            "backbone": spec["backbone"],
            "balance": spec["balance"],
            "outer_fold": outer_fold,
            "epoch": epoch,
            "train_loss": totals["loss"] / xtr.size(0),
            "train_outcome_loss": totals["outcome"] / xtr.size(0),
            "train_treatment_loss": totals["treatment"] / xtr.size(0),
            "train_balance_loss": totals["balance"] / xtr.size(0),
            "train_stability_loss": totals["stability"] / xtr.size(0),
            "valid_outcome_mse": float(valid_outcome),
            "valid_treatment_mse": float(valid_treatment),
            "valid_balance_loss": valid_balance,
            "active_balance_weight": active_balance_weight,
            "valid_score": float(valid_score),
        }
        history.append(row)
        print(
            f"[{spec['spec_id']} fold={outer_fold}] epoch={epoch:03d} "
            f"train={row['train_loss']:.4f} val={row['valid_score']:.4f} "
            f"y={row['valid_outcome_mse']:.4f} "
            f"d={row['valid_treatment_mse']:.4f} "
            f"bal={row['valid_balance_loss']:.4f}",
            flush=True,
        )

        if valid_score < best_score - args.min_delta:
            best_score = float(valid_score)
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(
            f"No valid state was produced for {spec['spec_id']} in outer fold {outer_fold}."
        )

    model.load_state_dict(best_state)
    model.to(device)
    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["valid_score"].idxmin()]
    metadata = {
        "spec_id": spec["spec_id"],
        "model": spec["name"],
        "outer_fold": outer_fold,
        "n_outer_train": int(len(X_train)),
        "n_encoder_fit": int(len(fit_idx)),
        "n_encoder_valid": int(len(valid_idx)),
        "best_epoch": int(best_row["epoch"]),
        "epochs_ran": int(len(history_df)),
        "best_valid_score": float(best_score),
        "best_valid_outcome_mse": float(best_row["valid_outcome_mse"]),
        "best_valid_treatment_mse": float(best_row["valid_treatment_mse"]),
        "best_valid_balance_loss": float(best_row["valid_balance_loss"]),
        "active_balance_weight": float(active_balance_weight),
    }
    return model, history_df, metadata


def fully_crossfitted_representation_dml(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    spec: dict[str, str],
    folds: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Estimate one representation specification with full outer cross-fitting."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n = len(Y)

    y_hat = np.zeros(n, dtype=np.float64)
    t_hat = np.zeros(n, dtype=np.float64)
    z_oof = np.full((n, args.latent_dim), np.nan, dtype=np.float32)
    fold_id = np.zeros(n, dtype=np.int64)
    histories: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    device = resolve_device(args.device)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            assert np.intersect1d(tr, te).size == 0
            model, history, encoder_meta = fit_encoder_on_outer_training_fold(
                X_train=X[tr],
                Y_train=Y[tr],
                T_train=T[tr],
                spec=spec,
                args=args,
                seed=seed + 10_000 * fold,
                outer_fold=fold,
            )
            histories.append(history)

            z_train = encode_model(
                model,
                X[tr],
                batch_size=args.encode_batch_size,
                device=device,
            )
            z_test = encode_model(
                model,
                X[te],
                batch_size=args.encode_batch_size,
                device=device,
            )
            z_oof[te] = z_test
            fold_id[te] = fold

            ym = LassoCV(
                cv=3,
                random_state=seed + fold,
                max_iter=10000,
                n_jobs=1,
            )
            tm = LassoCV(
                cv=3,
                random_state=seed + fold + 100,
                max_iter=10000,
                n_jobs=1,
            )
            ym.fit(z_train.astype(np.float64), Y[tr])
            tm.fit(z_train.astype(np.float64), T[tr])
            y_hat[te] = ym.predict(z_test.astype(np.float64))
            t_hat[te] = tm.predict(z_test.astype(np.float64))

            audits.append(
                {
                    "spec_id": spec["spec_id"],
                    "model": spec["name"],
                    "backbone": spec["backbone"],
                    "balance": spec["balance"],
                    "outer_fold": fold,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "train_test_overlap_n": int(np.intersect1d(tr, te).size),
                    "heldout_DY_used_for_encoder": False,
                    "best_epoch": encoder_meta["best_epoch"],
                    "epochs_ran": encoder_meta["epochs_ran"],
                    "best_valid_score": encoder_meta["best_valid_score"],
                    "best_valid_outcome_mse": encoder_meta["best_valid_outcome_mse"],
                    "best_valid_treatment_mse": encoder_meta["best_valid_treatment_mse"],
                    "best_valid_balance_loss": encoder_meta["best_valid_balance_loss"],
                    "y_alpha": float(ym.alpha_),
                    "t_alpha": float(tm.alpha_),
                }
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if np.isnan(z_oof).any():
        raise RuntimeError(f"Incomplete OOF representation for {spec['spec_id']}.")
    if np.any(fold_id == 0):
        raise RuntimeError(f"Incomplete fold assignment for {spec['spec_id']}.")

    result = residual_hc3(Y, T, y_hat, t_hat)
    return (
        result,
        y_hat,
        t_hat,
        z_oof,
        fold_id,
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(audits),
    )


def make_stability_summary(effect_table: pd.DataFrame) -> pd.DataFrame:
    rep = effect_table[effect_table["spec_id"].isin(SPEC_LIBRARY.keys())].copy()
    if rep.empty:
        return pd.DataFrame()

    ft_sinkhorn_rows = rep[rep["spec_id"] == "ft_sinkhorn"]
    ft_sinkhorn_estimate = float(ft_sinkhorn_rows.iloc[0]["estimate"]) if not ft_sinkhorn_rows.empty else np.nan
    rep["delta_vs_ft_sinkhorn"] = rep["estimate"] - ft_sinkhorn_estimate
    rep["abs_delta_vs_ft_sinkhorn"] = np.abs(rep["delta_vs_ft_sinkhorn"])

    effect_table.loc[rep.index, "delta_vs_ft_sinkhorn"] = rep["delta_vs_ft_sinkhorn"]
    effect_table.loc[rep.index, "abs_delta_vs_ft_sinkhorn"] = rep["abs_delta_vs_ft_sinkhorn"]

    summary = {
        "n_representation_specs": int(len(rep)),
        "n_positive": int((rep["estimate"] > 0).sum()),
        "n_negative": int((rep["estimate"] < 0).sum()),
        "n_significant_positive_5pct": int(
            ((rep["estimate"] > 0) & (rep["p_value"] < 0.05)).sum()
        ),
        "n_significant_negative_5pct": int(
            ((rep["estimate"] < 0) & (rep["p_value"] < 0.05)).sum()
        ),
        "all_same_sign": bool(rep["sign"].nunique() == 1),
        "estimate_mean": float(rep["estimate"].mean()),
        "estimate_sd": float(rep["estimate"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "estimate_min": float(rep["estimate"].min()),
        "estimate_max": float(rep["estimate"].max()),
        "estimate_range": float(rep["estimate"].max() - rep["estimate"].min()),
        "ft_sinkhorn_estimate": ft_sinkhorn_estimate,
    }
    return pd.DataFrame([summary])


def save_completed_spec(
    output_dir: Path,
    spec: dict[str, str],
    effect_row: dict[str, Any],
    history: pd.DataFrame,
    audit: pd.DataFrame,
    y_hat: np.ndarray,
    t_hat: np.ndarray,
    fold_id: np.ndarray,
    z_oof: np.ndarray,
    args: argparse.Namespace,
) -> None:
    spec_id = spec["spec_id"]
    pd.DataFrame([effect_row]).to_csv(
        output_dir / f"effect_{spec_id}.csv",
        index=False,
    )
    history.to_csv(output_dir / f"training_history_{spec_id}.csv", index=False)
    audit.to_csv(output_dir / f"fold_audit_{spec_id}.csv", index=False)
    pd.DataFrame(
        {
            "outer_fold": fold_id,
            "y_hat_oof": y_hat,
            "t_hat_oof": t_hat,
        }
    ).to_csv(output_dir / f"oof_nuisance_{spec_id}.csv", index=False)

    if args.save_oof_representations:
        np.savez_compressed(
            output_dir / f"z_oof_{spec_id}.npz",
            z_oof=z_oof,
            outer_fold=fold_id,
        )

    spec_metadata = {
        "script_version": SCRIPT_VERSION,
        "spec": spec,
        "seed": int(args.seed),
        "args": vars(args),
        "effect": effect_row,
        "warning": (
            "OOF representations come from fold-specific encoders. Do not interpret "
            "their concatenated coordinates as one globally aligned embedding space."
        ),
    }
    (output_dir / f"metadata_{spec_id}.json").write_text(
        json.dumps(spec_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_completed_spec(
    output_dir: Path,
    spec: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame] | None:
    spec_id = spec["spec_id"]
    effect_path = output_dir / f"effect_{spec_id}.csv"
    history_path = output_dir / f"training_history_{spec_id}.csv"
    audit_path = output_dir / f"fold_audit_{spec_id}.csv"
    if not (effect_path.exists() and history_path.exists() and audit_path.exists()):
        return None
    effect_row = pd.read_csv(effect_path).iloc[0].to_dict()
    history = pd.read_csv(history_path)
    audit = pd.read_csv(audit_path)
    return effect_row, history, audit


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if not 0.0 < args.inner_valid_fraction < 0.5:
        raise ValueError("--inner-valid-fraction must be between 0 and 0.5.")
    if args.treatment_groups < 2:
        raise ValueError("--treatment-groups must be at least 2.")
    if args.sinkhorn_epsilon <= 0:
        raise ValueError("--sinkhorn-epsilon must be positive.")
    if args.sinkhorn_iters < 1:
        raise ValueError("--sinkhorn-iters must be at least 1.")
    if not args.mmd_sigmas or any(value <= 0 for value in args.mmd_sigmas):
        raise ValueError("--mmd-sigmas must contain positive values.")


def main() -> None:
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}", flush=True)
    print(f"[SCRIPT PATH] {Path(__file__).resolve()}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--treatment-groups", type=int, default=4)
    parser.add_argument("--outcome-weight", type=float, default=1.0)
    parser.add_argument("--treatment-weight", type=float, default=0.30)
    parser.add_argument("--sinkhorn-weight", type=float, default=0.10)
    parser.add_argument("--mmd-weight", type=float, default=0.10)
    parser.add_argument("--stability-weight", type=float, default=0.02)
    parser.add_argument("--perturb-std", type=float, default=0.02)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.25)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--mmd-sigmas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--specs",
        nargs="+",
        choices=list(SPEC_LIBRARY.keys()),
        default=list(SPEC_LIBRARY.keys()),
        help="Specifications to run. Default: all four core specifications.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=OUT_DIRNAME,
        help="Run-specific output subfolder under ./outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-spec CSV files in the same output folder.",
    )
    parser.add_argument(
        "--save-oof-representations",
        action="store_true",
        help="Save compressed OOF Z arrays. They are fold-specific and not globally aligned.",
    )
    args = parser.parse_args()
    validate_args(args)

    requested_specs = [SPEC_LIBRARY[spec_id] for spec_id in args.specs]

    cfg = Config()
    cfg.seed = args.seed
    output_name = args.output_name.strip()
    if not output_name or output_name in {".", ".."}:
        raise ValueError("--output-name must be a non-empty folder name.")
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a folder name, not a path.")
    cfg.output_dir = ROOT / "outputs" / output_name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] device={device}, name={torch.cuda.get_device_name(device.index or 0)}",
            flush=True,
        )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(cfg.output_dir),
        "requested_specs": requested_specs,
        "args": vars(args),
        "design": {
            "x_interface": "bundle['x_all']",
            "same_outer_folds_across_specs": True,
            "representation_fully_crossfitted": True,
            "heldout_DY_used_for_encoder": False,
            "nuisance_model": "LassoCV(cv=3, max_iter=10000)",
            "covariance_estimator": "HC3",
        },
    }
    (cfg.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bundle = prepare_dataset(cfg)
    print(
        f"Loaded panel: total={len(bundle['df'])}, "
        f"train={len(bundle['train_df'])}, valid={len(bundle['valid_df'])}, "
        f"test={len(bundle['test_df'])}, x_dim={bundle['x_all'].shape[1]}",
        flush=True,
    )
    print("Feature source is preserved exactly: bundle['x_all']", flush=True)

    ols = ols_ate(bundle["x_all"], bundle["y_all"], bundle["t_all"])
    raw_dml = dml_ate_lasso_quiet(
        bundle["x_all"],
        bundle["y_all"],
        bundle["t_all"],
        folds=cfg.folds,
        seed=cfg.seed,
    )
    effect_rows: list[dict[str, Any]] = [
        summarize_effect("OLS", ols, "ols"),
        summarize_effect("Raw-X DML", raw_dml, "raw_x"),
    ]
    all_histories: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    model_detail_rows: list[dict[str, Any]] = []

    print(
        f"OLS done: estimate={ols['estimate']:.6f}, p={ols['p_value']:.6g}",
        flush=True,
    )
    print(
        f"Raw-X DML done: estimate={raw_dml['estimate']:.6f}, p={raw_dml['p_value']:.6g}",
        flush=True,
    )

    for spec in requested_specs:
        print(f"\n=== Running {spec['name']} ===", flush=True)

        completed = load_completed_spec(cfg.output_dir, spec) if args.resume else None
        if completed is not None:
            effect_row, history, audit = completed
            print(f"[RESUME] Loaded completed {spec['spec_id']} from output folder.", flush=True)
        else:
            (
                dml_result,
                y_hat,
                t_hat,
                z_oof,
                fold_id,
                history,
                audit,
            ) = fully_crossfitted_representation_dml(
                bundle["x_all"],
                bundle["y_all"],
                bundle["t_all"],
                spec=spec,
                folds=cfg.folds,
                seed=cfg.seed,
                args=args,
            )
            effect_row = summarize_effect(spec["name"], dml_result, spec["spec_id"])
            save_completed_spec(
                cfg.output_dir,
                spec,
                effect_row,
                history,
                audit,
                y_hat,
                t_hat,
                fold_id,
                z_oof,
                args,
            )
            print(
                f"{spec['name']} done: estimate={dml_result['estimate']:.6f}, "
                f"p={dml_result['p_value']:.6g}",
                flush=True,
            )

        effect_rows.append(effect_row)
        all_histories.append(history)
        all_audits.append(audit)

        model_detail_rows.append(
            {
                "spec_id": spec["spec_id"],
                "model": spec["name"],
                "backbone": spec["backbone"],
                "balance": spec["balance"],
                "feature_input": "bundle['x_all']",
                "outer_split": f"same random KFold for all specs, folds={cfg.folds}",
                "encoder_train_scope": "outer-training fold only",
                "encoder_validation_scope": "random subset of outer-training fold only",
                "heldout_DY_used_for_encoder": False,
                "nuisance_model": "LassoCV(cv=3, max_iter=10000)",
                "covariance_estimator": "HC3",
                "device": str(device),
                "latent_dim": args.latent_dim,
                "d_model": args.d_model if spec["backbone"] == "ft" else np.nan,
                "n_layers": args.n_layers if spec["backbone"] == "ft" else np.nan,
                "n_heads": args.n_heads if spec["backbone"] == "ft" else np.nan,
                "mlp_hidden_dim": args.mlp_hidden_dim if spec["backbone"] == "mlp" else np.nan,
                "balance_weight": balance_weight_for_spec(spec, args),
            }
        )

    effects = pd.DataFrame(effect_rows)
    for column in ["delta_vs_ft_sinkhorn", "abs_delta_vs_ft_sinkhorn"]:
        effects[column] = np.nan
    stability_summary = make_stability_summary(effects)

    model_details = pd.DataFrame(model_detail_rows)
    training_history = (
        pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    )
    fold_audit = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()

    sample_info = pd.DataFrame(
        [
            {
                "split": "all",
                "nobs": len(bundle["df"]),
                "x_dim": bundle["x_all"].shape[1],
                "z_dim": args.latent_dim,
            },
            {
                "split": "train",
                "nobs": len(bundle["train_df"]),
                "x_dim": bundle["x_train"].shape[1],
                "z_dim": args.latent_dim,
            },
            {
                "split": "valid",
                "nobs": len(bundle["valid_df"]),
                "x_dim": bundle["x_valid"].shape[1],
                "z_dim": args.latent_dim,
            },
            {
                "split": "test",
                "nobs": len(bundle["test_df"]),
                "x_dim": bundle["x_test"].shape[1],
                "z_dim": args.latent_dim,
            },
        ]
    )

    write_workbook(
        cfg.output_dir / "core_spec_stability_full_crossfit.xlsx",
        {
            "effects": effects,
            "stability_summary": stability_summary,
            "model_details": model_details,
            "sample_info": sample_info,
            "fold_audit": fold_audit,
            "training_history": training_history,
        },
        [
            "Representation-specification stability experiment.",
            "All representation specifications use the same outer folds, bundle['x_all'] input, LassoCV nuisance models, and HC3 inference.",
            "Each held-out observation is encoded by a fold-specific model trained and tuned without that observation's treatment or outcome.",
            "Only backbone and balance-loss type vary across the four core specifications.",
            "OOF representations are produced by different fold-specific encoders and must not be treated as one globally aligned embedding space.",
        ],
    )

    effects.to_csv(cfg.output_dir / "core_spec_effects.csv", index=False)
    stability_summary.to_csv(cfg.output_dir / "core_spec_stability_summary.csv", index=False)
    model_details.to_csv(cfg.output_dir / "model_details.csv", index=False)
    fold_audit.to_csv(cfg.output_dir / "fold_audit_all_specs.csv", index=False)
    training_history.to_csv(cfg.output_dir / "training_history_all_specs.csv", index=False)

    run_metadata = {
        **run_config,
        "data": {
            "search_file": str(cfg.search_path),
            "innovation_file": str(cfg.innovation_path),
            "outcome": cfg.outcome,
            "treatment": cfg.treatment,
            "controls": cfg.controls,
            "n_total": len(bundle["df"]),
            "n_train": len(bundle["train_df"]),
            "n_valid": len(bundle["valid_df"]),
            "n_test": len(bundle["test_df"]),
            "x_dim": int(bundle["x_all"].shape[1]),
            "z_dim": int(args.latent_dim),
        },
        "effects": effects.to_dict(orient="records"),
        "stability_summary": stability_summary.to_dict(orient="records"),
    }
    (cfg.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + effects.to_string(index=False))
    if not stability_summary.empty:
        print("\nStability summary:")
        print(stability_summary.to_string(index=False))
    print(f"Output directory: {cfg.output_dir}")


if __name__ == "__main__":
    main()
