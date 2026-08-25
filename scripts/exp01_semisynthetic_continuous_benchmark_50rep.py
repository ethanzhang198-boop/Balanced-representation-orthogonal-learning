from __future__ import annotations

"""Semi-synthetic continuous-treatment benchmark."""

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import ndtr
from scipy.stats import chi2, rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCRIPT_VERSION = "SEMISYNTHETIC_TRANSPORT_CORR045_50REP_20260810"
DEFAULT_OUTPUT_NAME = "exp01_semisynthetic_continuous_benchmark_50rep"
SCENARIO_ID = "semisynthetic_transport_corr045"

KAPPA = 0.90
TRANSPORT_STRENGTH = 0.75
CORR_G_H = 0.45

METHOD_LIBRARY: dict[str, str] = {
    "raw_x_dml": "Raw-X DML",
    "plain_ft": "Fully cross-fitted Plain FT-DML",
    "mlp_sinkhorn": "Fully cross-fitted BROL-MLP-Sinkhorn",
    "ft_mmd": "Fully cross-fitted BROL-FT-MMD",
    "ft_sinkhorn": "Fully cross-fitted BROL-FT-Sinkhorn",
    "gps": "Generalized Propensity Score",
    "drnet": "DRNet (self-contained implementation)",
    "vcnet": "VCNet (self-contained implementation)",
}

DEFAULT_METHODS = [
    "raw_x_dml",
    "plain_ft",
    "mlp_sinkhorn",
    "ft_mmd",
    "ft_sinkhorn",
    "gps",
    "drnet",
    "vcnet",
]

REPRESENTATION_METHODS = {"plain_ft", "mlp_sinkhorn", "ft_mmd", "ft_sinkhorn"}
INFERENCE_METHODS = {"raw_x_dml", *REPRESENTATION_METHODS}


def load_ablation_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "exp02_empirical_core_ablation_full_crossfit.py"
    if not path.exists():
        raise FileNotFoundError(
            "Could not find exp02_empirical_core_ablation_full_crossfit.py beside "
            "this benchmark script."
        )

    spec = importlib.util.spec_from_file_location("brol_core_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "Config",
        "prepare_dataset",
        "set_seed",
        "SPEC_LIBRARY",
        "dml_ate_lasso_quiet",
        "fully_crossfitted_representation_dml",
        "resolve_device",
        "write_workbook",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"The ablation module {path.name} is missing required objects: {missing}"
        )
    print(f"[CORE MODULE] {path}", flush=True)
    return module


CORE = load_ablation_module()


def safe_set_seed(seed: int) -> None:
    CORE.set_seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)

def append_result_row(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    atomic_write_csv(updated, path)

def normal_density(x: np.ndarray, mean: np.ndarray, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-6)
    z = (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / sd
    return np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sd)

def summarize_scalar_effect(
    estimate: float,
    true_tau: float,
    std_err: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> dict[str, Any]:
    estimate = float(estimate)
    error = estimate - float(true_tau)
    has_ci = (
        std_err is not None
        and ci_low is not None
        and ci_high is not None
        and np.isfinite(std_err)
        and np.isfinite(ci_low)
        and np.isfinite(ci_high)
    )
    return {
        "estimate": estimate,
        "std_err": float(std_err) if std_err is not None else np.nan,
        "ci_low": float(ci_low) if ci_low is not None else np.nan,
        "ci_high": float(ci_high) if ci_high is not None else np.nan,
        "error": float(error),
        "abs_error": float(abs(error)),
        "squared_error": float(error * error),
        "coverage_95": (
            bool(float(ci_low) <= true_tau <= float(ci_high)) if has_ci else np.nan
        ),
        "ci_length": float(ci_high - ci_low) if has_ci else np.nan,
        "sign_correct": bool(np.sign(estimate) == np.sign(true_tau)),
    }


@dataclass(frozen=True)
class DGPDesign:
    sample_indices: np.ndarray
    scenario_feature_indices: np.ndarray
    scenario_feature_names: list[str]
    x_mean: np.ndarray
    x_sd: np.ndarray


def infer_feature_names(bundle: dict[str, Any], p: int) -> list[str]:
    feature_name_keys = [
        "feature_names",
        "x_names",
        "control_names",
        "columns_x",
    ]
    for key in feature_name_keys:
        value = bundle.get(key)
        if value is not None and len(value) == p:
            return [str(v) for v in value]
    return [f"x_{j}" for j in range(p)]


def build_dgp_design(
    X_all: np.ndarray,
    feature_names: list[str],
    sample_size: int,
    scenario_seed: int,
) -> DGPDesign:
    X_all = np.asarray(X_all, dtype=np.float64)
    n, p = X_all.shape
    if sample_size <= 0 or sample_size >= n:
        sample_indices = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(scenario_seed)
        sample_indices = np.sort(rng.choice(n, size=sample_size, replace=False))

    X = X_all[sample_indices]
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    nonconstant = np.where(np.isfinite(sd) & (sd > 1e-8))[0]
    if len(nonconstant) < 16:
        raise ValueError(
            f"The semi-synthetic DGP requires at least 16 nonconstant X columns; found {len(nonconstant)}."
        )

    rng = np.random.default_rng(scenario_seed + 991)
    active = np.asarray(rng.permutation(nonconstant)[:16], dtype=np.int64)
    scenario_feature_names = [feature_names[j] for j in active]
    return DGPDesign(
        sample_indices=sample_indices,
        scenario_feature_indices=active,
        scenario_feature_names=scenario_feature_names,
        x_mean=mean,
        x_sd=np.where(sd > 1e-8, sd, 1.0),
    )


def standardized_design_matrix(X: np.ndarray, design: DGPDesign) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = (X - design.x_mean) / design.x_sd
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def standardized_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    sd = float(np.std(score, ddof=0))
    if sd < 1e-8:
        raise RuntimeError("Generated DGP score has near-zero variance.")
    return (score - float(np.mean(score))) / sd



def make_semisynthetic_sample(
    X_input: np.ndarray,
    design: DGPDesign,
    replication_seed: int,
    true_tau: float,
    confounding_scale: float,
    outcome_noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one semi-synthetic sample."""
    X_input = np.asarray(X_input, dtype=np.float64)
    X = X_input[design.sample_indices]
    Xs = standardized_design_matrix(X, design)
    a = design.scenario_feature_indices
    reference = np.asarray(Xs[:, a], dtype=np.float64)

    assignment_score = standardized_score(
        0.72 * reference[:, 0] * reference[:, 1]
        - 0.58 * np.sin(reference[:, 2])
        + 0.46 * reference[:, 3]
        + 0.38 * np.tanh(reference[:, 4] * reference[:, 5])
        - 0.30 * (reference[:, 6] ** 2 - 1.0)
        + 0.24 * reference[:, 7] * reference[:, 8]
    )
    group_cutoffs = np.quantile(assignment_score, [0.25, 0.50, 0.75])
    latent_group = np.digitize(assignment_score, group_cutoffs, right=True)
    group_centers = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    center = group_centers[latent_group]

    shared_geometry = standardized_score(
        0.56 * np.max(reference[:, :4], axis=1)
        - 0.48 * np.min(reference[:, 4:8], axis=1)
        + 0.42 * reference[:, 8] * reference[:, 9]
        - 0.34 * np.sin(reference[:, 10] * reference[:, 11])
        + 0.28 * reference[:, 12] * reference[:, 13] * reference[:, 14]
        + 0.22 * reference[:, 15]
    )
    g = standardized_score(
        0.92 * center + 0.34 * assignment_score + 0.22 * shared_geometry
    )
    h = standardized_score(
        0.68 * assignment_score
        + 0.62 * shared_geometry
        + 0.48 * center
        + 0.30 * reference[:, 0] * reference[:, 5]
    )

    transported = np.asarray(Xs, dtype=np.float64).copy()
    direction = np.linspace(-1.0, 1.0, len(a), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    shifts = (0, 3, 7, 11)
    for group_id in range(4):
        mask = latent_group == group_id
        mapped = np.roll(reference[mask], shift=shifts[group_id], axis=1)
        mapped = mapped + TRANSPORT_STRENGTH * group_centers[group_id] * direction
        transported[np.ix_(mask, a)] = mapped

    g_centered = g - float(np.mean(g))
    h_centered = h - float(np.mean(h))
    projection = float(
        np.dot(h_centered, g_centered)
        / max(np.dot(g_centered, g_centered), 1e-12)
    )
    h_orthogonal = standardized_score(h_centered - projection * g_centered)
    h = standardized_score(
        CORR_G_H * standardized_score(g_centered)
        + math.sqrt(1.0 - CORR_G_H ** 2) * h_orthogonal
    )

    conditional_treatment_mean = ndtr(
        KAPPA * g / math.sqrt(KAPPA * KAPPA + 2.0)
    )

    rng = np.random.default_rng(replication_seed)
    eps_d = rng.normal(0.0, 1.0, size=len(X))
    eps_y = float(outcome_noise_sd) * rng.normal(0.0, 1.0, size=len(X))

    latent_treatment = (KAPPA * g + eps_d) / math.sqrt(KAPPA * KAPPA + 1.0)
    treatment = np.clip(ndtr(latent_treatment), 1e-4, 1.0 - 1e-4)
    outcome = (
        float(true_tau) * treatment
        + float(confounding_scale) * h
        + eps_y
    )

    diagnostics = {
        "scenario": SCENARIO_ID,
        "replication_seed": int(replication_seed),
        "n": int(len(X)),
        "p": int(transported.shape[1]),
        "true_tau": float(true_tau),
        "scenario_type": "stratified_covariate_transport",
        "covariate_transform": "within-stratum permutation and shift",
        "transport_strength": float(TRANSPORT_STRENGTH),
        "corr_g_h_parameter": float(CORR_G_H),
        "noise_profile": "gaussian",
        "kappa": float(KAPPA),
        "confounding_scale": float(confounding_scale),
        "outcome_noise_sd": float(outcome_noise_sd),
        "treatment_mean": float(np.mean(treatment)),
        "treatment_sd": float(np.std(treatment, ddof=0)),
        "treatment_p01": float(np.quantile(treatment, 0.01)),
        "treatment_p05": float(np.quantile(treatment, 0.05)),
        "treatment_p50": float(np.quantile(treatment, 0.50)),
        "treatment_p95": float(np.quantile(treatment, 0.95)),
        "treatment_p99": float(np.quantile(treatment, 0.99)),
        "corr_treatment_g": float(np.corrcoef(treatment, g)[0, 1]),
        "corr_treatment_h": float(np.corrcoef(treatment, h)[0, 1]),
        "corr_g_h": float(np.corrcoef(g, h)[0, 1]),
        "corr_mu_t_h": float(np.corrcoef(conditional_treatment_mean, h)[0, 1]),
        "outcome_mean": float(np.mean(outcome)),
        "outcome_sd": float(np.std(outcome, ddof=0)),
    }
    return (
        transported.astype(np.float32),
        outcome.astype(np.float64),
        treatment.astype(np.float64),
        diagnostics,
    )


def fit_gps_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    folds: int,
    seed: int,
    grid_size: int,
) -> dict[str, Any]:
    """Estimate the GPS baseline."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n = len(T)
    mu_oof = np.zeros(n, dtype=np.float64)
    alphas = np.logspace(-4, 4, 25)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            model = RidgeCV(alphas=alphas, cv=3)
            model.fit(X[tr], T[tr])
            mu_oof[te] = model.predict(X[te])

    sigma = float(np.sqrt(np.mean((T - mu_oof) ** 2)))
    sigma = max(sigma, 1e-4)
    gps_observed = normal_density(T, mu_oof, sigma)

    response_design = np.column_stack(
        [
            T,
            T ** 2,
            gps_observed,
            gps_observed ** 2,
            T * gps_observed,
        ]
    )
    fit = sm.OLS(Y, sm.add_constant(response_design, has_constant="add")).fit(cov_type="HC3")

    treatment_model = RidgeCV(alphas=alphas, cv=3)
    treatment_model.fit(X, T)
    mu_full = treatment_model.predict(X)

    lo = float(np.quantile(T, 0.05))
    hi = float(np.quantile(T, 0.95))
    treatment_grid = np.linspace(lo, hi, int(grid_size))
    adrf = []
    for d in treatment_grid:
        d_vec = np.full(n, d, dtype=np.float64)
        gps_d = normal_density(d_vec, mu_full, sigma)
        design_d = np.column_stack(
            [
                d_vec,
                d_vec ** 2,
                gps_d,
                gps_d ** 2,
                d_vec * gps_d,
            ]
        )
        pred = fit.predict(sm.add_constant(design_d, has_constant="add"))
        adrf.append(float(np.mean(pred)))

    slope = float(LinearRegression().fit(treatment_grid.reshape(-1, 1), np.asarray(adrf)).coef_[0])
    return {
        "estimate": slope,
        "std_err": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "gps_sigma": sigma,
        "adrf_grid_low": lo,
        "adrf_grid_high": hi,
        "adrf_grid_size": int(grid_size),
    }

class TruncatedPowerBasis(nn.Module):
    def __init__(self, degree: int = 2, knots: Iterable[float] = (0.25, 0.50, 0.75)):
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = int(degree)
        self.register_buffer("knots", torch.tensor(list(knots), dtype=torch.float32))
        self.n_basis = self.degree + 1 + len(list(knots))

    def forward(self, treatment: torch.Tensor) -> torch.Tensor:
        t = treatment.reshape(-1, 1)
        pieces = [torch.ones_like(t)]
        for power in range(1, self.degree + 1):
            pieces.append(t ** power)
        for knot in self.knots:
            pieces.append(F.relu(t - knot) ** self.degree)
        return torch.cat(pieces, dim=1)


class DynamicLinear(nn.Module):
    """Treatment-indexed varying-coefficient linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: TruncatedPowerBasis,
        activation: str | None,
        append_treatment: bool,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.basis = basis
        self.append_treatment = bool(append_treatment)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, basis.n_basis)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, basis.n_basis))
        nn.init.xavier_uniform_(self.weight.reshape(out_features, -1))
        if activation == "relu":
            self.activation: nn.Module | None = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, treatment: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b = self.basis(treatment)  # [batch, basis]
        weight_t = torch.einsum("bk,oik->boi", b, self.weight)
        bias_t = torch.einsum("bk,ok->bo", b, self.bias)
        out = torch.bmm(weight_t, features.unsqueeze(-1)).squeeze(-1) + bias_t
        if self.activation is not None:
            out = self.activation(out)
        if self.append_treatment:
            out = torch.cat([treatment.reshape(-1, 1), out], dim=1)
        return out


class VCNetBaseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        dynamic_dim: int = 64,
        density_grid: int = 20,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.density_grid = int(density_grid)
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.density_head = nn.Linear(hidden_dim, self.density_grid + 1)
        basis1 = TruncatedPowerBasis(degree=2, knots=(0.25, 0.50, 0.75))
        basis2 = TruncatedPowerBasis(degree=2, knots=(0.25, 0.50, 0.75))
        self.dynamic1 = DynamicLinear(
            hidden_dim,
            dynamic_dim,
            basis=basis1,
            activation="relu",
            append_treatment=False,
        )
        self.dynamic2 = DynamicLinear(
            dynamic_dim,
            1,
            basis=basis2,
            activation=None,
            append_treatment=False,
        )

    def density_at_treatment(self, treatment: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.density_head(hidden)
        probs = torch.softmax(logits, dim=1)
        scaled = torch.clamp(treatment, 0.0, 1.0) * self.density_grid
        lower = torch.floor(scaled).long().clamp(0, self.density_grid)
        upper = torch.ceil(scaled).long().clamp(0, self.density_grid)
        fraction = scaled - lower.float()
        p_lower = probs.gather(1, lower.reshape(-1, 1)).squeeze(1)
        p_upper = probs.gather(1, upper.reshape(-1, 1)).squeeze(1)
        return (1.0 - fraction) * p_lower + fraction * p_upper

    def outcome(self, treatment: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        q1 = self.dynamic1(treatment, hidden)
        return self.dynamic2(treatment, q1).reshape(-1)

    def forward(self, treatment: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.feature_net(x)
        density = self.density_at_treatment(treatment, hidden)
        outcome = self.outcome(treatment, hidden)
        return density, outcome


def vcnet_predict_adrf(
    model: VCNetBaseline,
    X: np.ndarray,
    grid: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    adrf: list[float] = []
    with torch.no_grad():
        for d in grid:
            predictions: list[np.ndarray] = []
            for start in range(0, len(X), batch_size):
                bx = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
                bt = torch.full((bx.size(0),), float(d), dtype=torch.float32, device=device)
                hidden = model.feature_net(bx)
                pred = model.outcome(bt, hidden)
                predictions.append(pred.detach().cpu().numpy())
            adrf.append(float(np.mean(np.concatenate(predictions))))
    return np.asarray(adrf, dtype=np.float64)


def fit_vcnet_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    idx = np.arange(len(X))
    train_idx, valid_idx = train_test_split(
        idx,
        test_size=args.inner_valid_fraction,
        random_state=seed,
        shuffle=True,
    )

    safe_set_seed(seed)
    device = CORE.resolve_device(args.device)
    model = VCNetBaseline(
        n_features=X.shape[1],
        hidden_dim=args.vcnet_hidden_dim,
        dynamic_dim=args.vcnet_dynamic_dim,
        density_grid=args.vcnet_density_grid,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xtr = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(Y[train_idx], dtype=torch.float32, device=device)
    ttr = torch.tensor(T[train_idx], dtype=torch.float32, device=device)
    xva = torch.tensor(X[valid_idx], dtype=torch.float32, device=device)
    yva = torch.tensor(Y[valid_idx], dtype=torch.float32, device=device)
    tva = torch.tensor(T[valid_idx], dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    wait = 0
    n_batches = math.ceil(len(train_idx) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(xtr.size(0), device=device)
        train_total = 0.0
        for batch in range(n_batches):
            bidx = order[batch * args.batch_size : (batch + 1) * args.batch_size]
            bx, by, bt = xtr[bidx], ytr[bidx], ttr[bidx]
            optimizer.zero_grad(set_to_none=True)
            density, outcome = model(bt, bx)
            outcome_loss = F.mse_loss(outcome, by)
            density_loss = -torch.log(density.clamp_min(1e-8)).mean()
            loss = outcome_loss + args.vcnet_density_weight * density_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * bx.size(0)

        model.eval()
        with torch.no_grad():
            density_val, outcome_val = model(tva, xva)
            val_outcome = float(F.mse_loss(outcome_val, yva).item())
            val_density = float((-torch.log(density_val.clamp_min(1e-8))).mean().item())
            val_score = val_outcome + args.vcnet_density_weight * val_density

        if args.verbose_training:
            print(
                f"[vcnet] epoch={epoch:03d} train={train_total / len(train_idx):.5f} "
                f"val={val_score:.5f} y={val_outcome:.5f} density={val_density:.5f}",
                flush=True,
            )

        if val_score < best_score - args.min_delta:
            best_score = val_score
            best_epoch = epoch
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("VCNet training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    grid = np.linspace(
        float(np.quantile(T, 0.05)),
        float(np.quantile(T, 0.95)),
        args.adrf_grid_size,
    )
    adrf = vcnet_predict_adrf(
        model,
        X,
        grid,
        batch_size=args.encode_batch_size,
        device=device,
    )
    slope = float(LinearRegression().fit(grid.reshape(-1, 1), adrf).coef_[0])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "estimate": slope,
        "std_err": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "adrf_grid_low": float(grid[0]),
        "adrf_grid_high": float(grid[-1]),
        "adrf_grid_size": int(len(grid)),
    }

class DRNetBaseline(nn.Module):
    """Single-treatment DRNet baseline."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_strata: int = 5,
        dropout: float = 0.10,
    ):
        super().__init__()
        if n_strata < 2:
            raise ValueError("DRNet requires at least two dosage strata.")
        self.n_strata = int(n_strata)
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        head_hidden = max(hidden_dim // 2, 16)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, head_hidden),
                    nn.ELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(self.n_strata)
            ]
        )

    def stratum_index(self, treatment: torch.Tensor) -> torch.Tensor:
        return torch.floor(
            torch.clamp(treatment, 0.0, 1.0 - 1e-7) * self.n_strata
        ).long()

    def outcome(
        self,
        treatment: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([hidden, treatment.reshape(-1, 1)], dim=1)
        all_heads = torch.cat([head(inputs) for head in self.heads], dim=1)
        index = self.stratum_index(treatment).reshape(-1, 1)
        return all_heads.gather(1, index).reshape(-1)

    def forward(self, treatment: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.outcome(treatment, self.feature_net(x))


def drnet_predict_adrf(
    model: DRNetBaseline,
    X: np.ndarray,
    grid: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    adrf: list[float] = []
    with torch.no_grad():
        for d in grid:
            predictions: list[np.ndarray] = []
            for start in range(0, len(X), batch_size):
                bx = torch.tensor(
                    X[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                bt = torch.full(
                    (bx.size(0),),
                    float(d),
                    dtype=torch.float32,
                    device=device,
                )
                predictions.append(model(bt, bx).detach().cpu().numpy())
            adrf.append(float(np.mean(np.concatenate(predictions))))
    return np.asarray(adrf, dtype=np.float64)


def fit_drnet_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    idx = np.arange(len(X))
    train_idx, valid_idx = train_test_split(
        idx,
        test_size=args.inner_valid_fraction,
        random_state=seed,
        shuffle=True,
    )

    x_mean = np.mean(X[train_idx], axis=0, dtype=np.float64)
    x_sd = np.std(X[train_idx], axis=0, ddof=0, dtype=np.float64)
    x_sd = np.where(x_sd > 1e-8, x_sd, 1.0)
    X_scaled = ((X - x_mean) / x_sd).astype(np.float32)

    safe_set_seed(seed)
    device = CORE.resolve_device(args.device)
    model = DRNetBaseline(
        n_features=X.shape[1],
        hidden_dim=args.drnet_hidden_dim,
        n_strata=args.drnet_strata,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    xtr = torch.tensor(X_scaled[train_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(Y[train_idx], dtype=torch.float32, device=device)
    ttr = torch.tensor(T[train_idx], dtype=torch.float32, device=device)
    xva = torch.tensor(X_scaled[valid_idx], dtype=torch.float32, device=device)
    yva = torch.tensor(Y[valid_idx], dtype=torch.float32, device=device)
    tva = torch.tensor(T[valid_idx], dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    wait = 0
    n_batches = math.ceil(len(train_idx) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(xtr.size(0), device=device)
        train_total = 0.0
        for batch in range(n_batches):
            bidx = order[
                batch * args.batch_size : (batch + 1) * args.batch_size
            ]
            bx, by, bt = xtr[bidx], ytr[bidx], ttr[bidx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(bt, bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * bx.size(0)

        model.eval()
        with torch.no_grad():
            val_score = float(F.mse_loss(model(tva, xva), yva).item())
        if args.verbose_training:
            print(
                f"[drnet] epoch={epoch:03d} "
                f"train={train_total / len(train_idx):.5f} "
                f"val={val_score:.5f}",
                flush=True,
            )
        if val_score < best_score - args.min_delta:
            best_score = val_score
            best_epoch = epoch
            best_state = deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("DRNet training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    grid = np.linspace(
        float(np.quantile(T, 0.05)),
        float(np.quantile(T, 0.95)),
        args.adrf_grid_size,
    )
    adrf = drnet_predict_adrf(
        model,
        X_scaled,
        grid,
        batch_size=args.encode_batch_size,
        device=device,
    )
    slope = float(
        LinearRegression().fit(grid.reshape(-1, 1), adrf).coef_[0]
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "estimate": slope,
        "std_err": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "drnet_strata": int(args.drnet_strata),
        "adrf_grid_low": float(grid[0]),
        "adrf_grid_high": float(grid[-1]),
        "adrf_grid_size": int(len(grid)),
    }


def run_method(
    method_id: str,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    replication_seed: int,
    true_tau: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()

    if method_id == "raw_x_dml":
        result = CORE.dml_ate_lasso_quiet(
            X,
            Y,
            T,
            folds=args.folds,
            seed=replication_seed,
        )
    elif method_id in REPRESENTATION_METHODS:
        spec = CORE.SPEC_LIBRARY[method_id]
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            result, _, _, _, _, _, _ = CORE.fully_crossfitted_representation_dml(
                X,
                Y,
                T,
                spec=spec,
                folds=args.folds,
                seed=replication_seed,
                args=args,
            )
    elif method_id == "gps":
        result = fit_gps_baseline(
            X,
            Y,
            T,
            folds=args.folds,
            seed=replication_seed,
            grid_size=args.adrf_grid_size,
        )
    elif method_id == "drnet":
        result = fit_drnet_baseline(
            X,
            Y,
            T,
            seed=replication_seed + 600_000,
            args=args,
        )
    elif method_id == "vcnet":
        result = fit_vcnet_baseline(
            X,
            Y,
            T,
            seed=replication_seed + 700_000,
            args=args,
        )
    else:
        raise KeyError(f"Unknown method: {method_id}")

    summary = summarize_scalar_effect(
        estimate=float(result["estimate"]),
        true_tau=true_tau,
        std_err=result.get("std_err"),
        ci_low=result.get("ci_low"),
        ci_high=result.get("ci_high"),
    )
    extras = {
        key: value
        for key, value in result.items()
        if key not in {"estimate", "std_err", "ci_low", "ci_high"}
        and np.isscalar(value)
    }
    return {
        **summary,
        **extras,
        "runtime_seconds": float(time.time() - started),
    }


def boolean_series_to_float(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    mapping = {
        True: 1.0,
        False: 0.0,
        "True": 1.0,
        "False": 0.0,
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        1: 1.0,
        0: 0.0,
    }
    return series.map(mapping).astype(float)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (scenario, method_id, method), group in ok.groupby(
        ["scenario", "method_id", "method"], sort=False
    ):
        estimate = pd.to_numeric(group["estimate"], errors="coerce")
        error = pd.to_numeric(group["error"], errors="coerce")
        std_err = pd.to_numeric(group.get("std_err"), errors="coerce")
        coverage = boolean_series_to_float(group.get("coverage_95"))
        ci_length = pd.to_numeric(group.get("ci_length"), errors="coerce")
        runtime = pd.to_numeric(group.get("runtime_seconds"), errors="coerce")
        sign_correct = boolean_series_to_float(group.get("sign_correct"))
        rows.append(
            {
                "scenario": scenario,
                "method_id": method_id,
                "method": method,
                "n_success": int(len(group)),
                "estimate_mean": float(estimate.mean()),
                "estimate_median": float(estimate.median()),
                "bias": float(error.mean()),
                "absolute_bias_of_mean": float(abs(error.mean())),
                "mean_absolute_error": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "empirical_sd": float(estimate.std(ddof=1)) if len(group) > 1 else np.nan,
                "mean_estimated_se": float(std_err.mean()) if std_err.notna().any() else np.nan,
                "coverage_95": float(coverage.mean()) if coverage.notna().any() else np.nan,
                "average_ci_length": float(ci_length.mean()) if ci_length.notna().any() else np.nan,
                "sign_accuracy": float(sign_correct.mean()) if sign_correct.notna().any() else np.nan,
                "mean_runtime_seconds": float(runtime.mean()),
            }
        )
    aggregate = pd.DataFrame(rows)
    if aggregate.empty:
        return aggregate

    rank_specs = {
        "rank_absolute_bias_of_mean": "absolute_bias_of_mean",
        "rank_mean_absolute_error": "mean_absolute_error",
        "rank_rmse": "rmse",
    }
    for rank_column, metric_column in rank_specs.items():
        aggregate[rank_column] = (
            aggregate.groupby("scenario")[metric_column]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )
    return aggregate


def friedman_aligned_rank_postprocessing(
    results: pd.DataFrame, method_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the aligned-rank omnibus test from replication-level absolute errors."""
    columns = [
        "status",
        "performance_metric",
        "n_methods",
        "n_complete_replications",
        "n_excluded_replications",
        "statistic",
        "df",
        "p_value",
    ]
    rank_table = pd.DataFrame(
        {
            "method_id": method_order,
            "method": [METHOD_LIBRARY[method_id] for method_id in method_order],
            "mean_aligned_rank": np.nan,
        }
    )
    if len(method_order) < 2:
        return (
            pd.DataFrame(
                [["not_computed", "absolute_estimation_error", len(method_order), 0, 0, np.nan, np.nan, np.nan]],
                columns=columns,
            ),
            rank_table,
        )

    replication = pd.to_numeric(results.get("replication"), errors="coerce")
    all_replications = sorted(replication.dropna().astype(int).unique())
    ok = results.loc[results["status"] == "ok", ["replication", "method_id", "abs_error"]].copy()
    ok["replication"] = pd.to_numeric(ok["replication"], errors="coerce")
    ok["abs_error"] = pd.to_numeric(ok["abs_error"], errors="coerce")
    ok = ok.dropna(subset=["replication", "method_id", "abs_error"])
    ok["replication"] = ok["replication"].astype(int)
    if ok.duplicated(["replication", "method_id"]).any():
        raise ValueError("Aligned-rank post-processing requires one successful result per replication and method.")

    matrix = ok.pivot(index="replication", columns="method_id", values="abs_error").reindex(
        columns=method_order
    )
    complete = matrix.dropna(axis=0, how="any")
    n_complete = len(complete)
    n_excluded = len(all_replications) - n_complete
    if n_complete < 2:
        return (
            pd.DataFrame(
                [["not_computed", "absolute_estimation_error", len(method_order), n_complete, n_excluded, np.nan, np.nan, np.nan]],
                columns=columns,
            ),
            rank_table,
        )

    values = complete.to_numpy(dtype=float)
    aligned = values - values.mean(axis=1, keepdims=True)
    ranks = rankdata(aligned.ravel(), method="average").reshape(aligned.shape)
    grand_mean_rank = (ranks.size + 1.0) / 2.0
    rank_sums = ranks.sum(axis=0)
    denominator = float(np.square(ranks - grand_mean_rank).sum())
    if denominator <= 0.0:
        raise ValueError("Aligned-rank post-processing is undefined because all aligned values are tied.")
    statistic = float(
        (len(method_order) - 1)
        * np.square(rank_sums - n_complete * grand_mean_rank).sum()
        / denominator
    )
    df = len(method_order) - 1
    rank_table["mean_aligned_rank"] = ranks.mean(axis=0)
    return (
        pd.DataFrame(
            [["computed", "absolute_estimation_error", len(method_order), n_complete, n_excluded, statistic, df, float(chi2.sf(statistic, df))]],
            columns=columns,
        ),
        rank_table,
    )


def failure_summary(results: pd.DataFrame) -> pd.DataFrame:
    failed = results[results["status"] != "ok"].copy()
    if failed.empty:
        return pd.DataFrame(columns=["scenario", "method_id", "method", "n_failed"])
    return (
        failed.groupby(["scenario", "method_id", "method"], as_index=False)
        .size()
        .rename(columns={"size": "n_failed"})
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.epochs != 100:
        print(
            f"[NOTICE] The requested maximum epoch rule is 100, but --epochs={args.epochs} was supplied.",
            flush=True,
        )
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not 0.0 < args.inner_valid_fraction < 0.5:
        raise ValueError("--inner-valid-fraction must lie between 0 and 0.5")
    if args.true_tau == 0:
        raise ValueError("--true-tau should be nonzero so sign accuracy is meaningful")
    if args.adrf_grid_size < 5:
        raise ValueError("--adrf-grid-size must be at least 5")
    if args.drnet_strata < 2:
        raise ValueError("--drnet-strata must be at least 2")
    if args.drnet_hidden_dim < 8:
        raise ValueError("--drnet-hidden-dim must be at least 8")
    if args.sinkhorn_epsilon <= 0:
        raise ValueError("--sinkhorn-epsilon must be positive")
    if args.sinkhorn_iters < 1:
        raise ValueError("--sinkhorn-iters must be at least 1")
    if not args.mmd_sigmas or any(float(v) <= 0 for v in args.mmd_sigmas):
        raise ValueError("--mmd-sigmas must contain positive values")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="50-rep semi-synthetic continuous-treatment benchmark."
    )
    parser.add_argument("--seed", type=int, default=60000000)
    parser.add_argument("--scenario-seed", type=int, default=20260724)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--true-tau", type=float, default=0.20)
    parser.add_argument("--confounding-scale", type=float, default=0.10)
    parser.add_argument("--outcome-noise-sd", type=float, default=0.20)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_LIBRARY.keys()),
        default=DEFAULT_METHODS,
        help="Subset of the eight benchmark methods to run.",
    )

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
        "--mmd-sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0]
    )

    parser.add_argument("--adrf-grid-size", type=int, default=41)
    parser.add_argument("--drnet-hidden-dim", type=int, default=64)
    parser.add_argument("--drnet-strata", type=int, default=5)
    parser.add_argument("--vcnet-hidden-dim", type=int, default=64)
    parser.add_argument("--vcnet-dynamic-dim", type=int, default=64)
    parser.add_argument("--vcnet-density-grid", type=int, default=20)
    parser.add_argument("--vcnet-density-weight", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip replication/method cells already completed successfully.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}", flush=True)
    print(f"[SCRIPT PATH] {Path(__file__).resolve()}", flush=True)
    print(f"[EPOCH RULE] maximum={args.epochs}, patience={args.patience}", flush=True)

    cfg = CORE.Config()
    cfg.seed = int(args.seed)
    output_name = args.output_name.strip()
    if not output_name or output_name in {".", ".."} or Path(output_name).name != output_name:
        raise ValueError("--output-name must be a non-empty folder name, not a path")
    output_dir = PROJECT_ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_set_seed(args.seed)
    device = CORE.resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] device={device}, name={torch.cuda.get_device_name(device.index or 0)}",
            flush=True,
        )

    bundle = CORE.prepare_dataset(cfg)
    X_all = np.asarray(bundle["x_all"], dtype=np.float64)
    feature_names = infer_feature_names(bundle, X_all.shape[1])
    design = build_dgp_design(
        X_all,
        feature_names=feature_names,
        sample_size=args.sample_size,
        scenario_seed=args.scenario_seed,
    )

    run_config = {
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "args": vars(args),
        "scenario": {
            "id": SCENARIO_ID,
            "scenario_type": "stratified_covariate_transport",
            "covariate_transform": "within-stratum permutation and shift",
            "transport_strength": TRANSPORT_STRENGTH,
            "corr_g_h_parameter": CORR_G_H,
            "kappa": KAPPA,
        },
        "methods": {key: METHOD_LIBRARY[key] for key in args.methods},
        "data_generating_process": {
            "observed_reference_dataset_used": True,
            "sample_size": int(len(design.sample_indices)),
            "sample_indices_saved_to": "dgp_sample_indices.npy",
            "scenario_feature_indices": design.scenario_feature_indices.tolist(),
            "scenario_feature_names": design.scenario_feature_names,
            "treatment_equation": "D=Phi((kappa*g(X)+eps_D)/sqrt(kappa^2+1))",
            "outcome_equation": "Y=true_tau*D+confounding_scale*h(X)+eps_Y",
            "constant_marginal_effect": float(args.true_tau),
        },
        "design_guarantees": {
            "same_X_rows_across_replications": True,
            "same_data_for_all_methods_within_replication": True,
            "same_outer_folds_across_representation_specs_within_replication": True,
            "representation_fully_crossfitted": True,
            "heldout_DY_used_for_encoder": False,
            "maximum_epochs": int(args.epochs),
            "inner_early_stopping": True,
        },
        "vcnet_note": (
            "Self-contained varying-coefficient implementation with the "
            "additional regularization setting disabled."
        ),
        "drnet_note": (
            "Self-contained PyTorch implementation with shared covariate layers and "
            "equal-width dosage-stratum outcome heads."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(output_dir / "dgp_sample_indices.npy", design.sample_indices)
    pd.DataFrame(
        {
            "scenario_order": np.arange(len(design.scenario_feature_indices)),
            "feature_index": design.scenario_feature_indices,
            "feature_name": design.scenario_feature_names,
        }
    ).to_csv(output_dir / "scenario_features.csv", index=False)

    results_path = output_dir / "replication_results.csv"
    diagnostics_path = output_dir / "dgp_diagnostics.csv"
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    completed: set[tuple[int, str]] = set()
    if args.resume and not existing.empty:
        ok = existing[existing["status"] == "ok"]
        completed = {
            (int(row.replication), str(row.method_id)) for row in ok.itertuples()
        }
        print(f"[RESUME] {len(completed)} completed cells detected.", flush=True)

    total_cells = args.reps * len(args.methods)
    current_cell = 0
    for replication in range(1, args.reps + 1):
        replication_seed = int(args.seed) + 10_000 * replication
        X, Y, T, diagnostics = make_semisynthetic_sample(
            X_all,
            design=design,
            replication_seed=replication_seed,
            true_tau=args.true_tau,
            confounding_scale=args.confounding_scale,
            outcome_noise_sd=args.outcome_noise_sd,
        )
        diagnostics_row = {"replication": replication, **diagnostics}
        if diagnostics_path.exists():
            d_existing = pd.read_csv(diagnostics_path)
            mask = pd.to_numeric(d_existing["replication"], errors="coerce") == replication
            if not mask.any():
                atomic_write_csv(
                    pd.concat([d_existing, pd.DataFrame([diagnostics_row])], ignore_index=True),
                    diagnostics_path,
                )
        else:
            atomic_write_csv(pd.DataFrame([diagnostics_row]), diagnostics_path)

        print(
            f"\n=== replication={replication}/{args.reps} seed={replication_seed} n={len(Y)} ===",
            flush=True,
        )
        for method_id in args.methods:
            current_cell += 1
            key = (replication, method_id)
            if key in completed:
                print(
                    f"[{current_cell}/{total_cells}] [SKIP] rep={replication} {method_id}",
                    flush=True,
                )
                continue

            method_seed = replication_seed
            print(
                f"[{current_cell}/{total_cells}] Running {METHOD_LIBRARY[method_id]}...",
                flush=True,
            )
            base_row: dict[str, Any] = {
                "script_version": SCRIPT_VERSION,
                "scenario": SCENARIO_ID,
                "replication": int(replication),
                "replication_seed": int(replication_seed),
                "method_seed": int(method_seed),
                "method_id": method_id,
                "method": METHOD_LIBRARY[method_id],
                "n": int(len(Y)),
                "p": int(X.shape[1]),
                "true_tau": float(args.true_tau),
            }
            try:
                method_result = run_method(
                    method_id,
                    X,
                    Y,
                    T,
                    replication_seed=method_seed,
                    true_tau=args.true_tau,
                    args=args,
                )
                row = {
                    **base_row,
                    "status": "ok",
                    "error_type": "",
                    "error_message": "",
                    **method_result,
                }
                print(
                    f"[DONE] {method_id}: estimate={row['estimate']:.6f} "
                    f"error={row['error']:.6f} runtime={row['runtime_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    **base_row,
                    "status": "failed",
                    "estimate": np.nan,
                    "std_err": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "error": np.nan,
                    "abs_error": np.nan,
                    "squared_error": np.nan,
                    "coverage_95": np.nan,
                    "ci_length": np.nan,
                    "sign_correct": np.nan,
                    "runtime_seconds": np.nan,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                print(f"[FAILED] {method_id}: {type(exc).__name__}: {exc}", flush=True)
                with (output_dir / "failures.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"\n--- rep={replication} method={method_id} ---\n")
                    fh.write(traceback.format_exc())
            append_result_row(results_path, row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = pd.read_csv(results_path)
    aggregate = aggregate_results(results)
    friedman_aligned_rank, mean_aligned_ranks = friedman_aligned_rank_postprocessing(
        results, args.methods
    )
    failures = failure_summary(results)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    method_details = pd.DataFrame(
        [
            {
                "method_id": method_id,
                "method": METHOD_LIBRARY[method_id],
                "reports_analytic_ci": method_id in INFERENCE_METHODS,
                "fully_crossfitted_representation": method_id in REPRESENTATION_METHODS,
                "notes": (
                    "Point-estimate ADRF slope; no analytic CI claimed."
                    if method_id in {"gps", "drnet", "vcnet"}
                    else "Known-tau scalar effect with HC3 inference."
                ),
            }
            for method_id in args.methods
        ]
    )

    aggregate.to_csv(output_dir / "benchmark_summary.csv", index=False)
    friedman_aligned_rank.to_csv(output_dir / "friedman_aligned_rank_summary.csv", index=False)
    mean_aligned_ranks.to_csv(output_dir / "friedman_aligned_rank_mean_ranks.csv", index=False)
    failures.to_csv(output_dir / "failure_summary.csv", index=False)

    workbook_sheets = {
        "summary": aggregate,
        "friedman_aligned_rank": friedman_aligned_rank,
        "mean_aligned_ranks": mean_aligned_ranks,
        "replications": results,
        "dgp_diagnostics": diagnostics,
        "method_details": method_details,
        "scenario_features": pd.read_csv(output_dir / "scenario_features.csv"),
    }
    workbook_notes = [
            "Semi-synthetic continuous-treatment benchmark.",
        "Observed covariates are retained while treatment and outcome are generated.",
        f"The true constant marginal treatment effect is tau={args.true_tau}.",
        "Treatment and outcome use a shared stratum-wise covariate-transport scenario with corr(g,h)=0.45.",
        "All representation methods use fully cross-fitted fold-specific encoders.",
        f"This run uses {args.reps} Monte Carlo replications and {len(args.methods)} comparison methods.",
        f"The neural training budget is at most {args.epochs} epochs with inner early stopping.",
        "GPS, DRNet, and VCNet are compared on point-estimation bias and RMSE; no analytic CI is claimed for them.",
    ]
    CORE.write_workbook(
        output_dir / "continuous_treatment_benchmark.xlsx",
        workbook_sheets,
        workbook_notes,
    )

    run_metadata = {
        **run_config,
        "n_result_rows": int(len(results)),
        "n_success": int((results["status"] == "ok").sum()),
        "n_failed": int((results["status"] != "ok").sum()),
        "summary": aggregate.to_dict(orient="records"),
        "friedman_aligned_rank": friedman_aligned_rank.to_dict(orient="records"),
        "mean_aligned_ranks": mean_aligned_ranks.to_dict(orient="records"),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SUMMARY ===", flush=True)
    if aggregate.empty:
        print("No successful results.", flush=True)
    else:
        print(aggregate.to_string(index=False), flush=True)
    print(f"\nOutput directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
