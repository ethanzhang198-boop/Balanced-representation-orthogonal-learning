from __future__ import annotations

"""Orthogonal restricted-cubic-spline dose-response diagnostic."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import statsmodels.api as sm
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


SCRIPT_VERSION = "ORTHOGONAL_RCS_DOSE_RESPONSE_SEED42_20260801"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT / "outputs" / "exp05_orthogonal_rcs_dose_response"
OUTPUT_DIR = WORK_ROOT / "dose_response" / "seed42_orthogonal_rcs"
CORE_SCRIPT = PROJECT_ROOT / "scripts" / "exp02_empirical_core_ablation_full_crossfit.py"
CORE_LINEAR_EFFECT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp02_empirical_core_ablation_full_crossfit"
    / "effect_ft_sinkhorn.csv"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data_utils import prepare_dataset, set_seed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_core() -> Any:
    if not CORE_SCRIPT.exists():
        raise FileNotFoundError(CORE_SCRIPT)
    spec = importlib.util.spec_from_file_location("rcs_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def restricted_cubic_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell-style RCS basis: one linear and K-2 nonlinear columns."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if len(knots) < 4 or not np.all(np.diff(knots) > 0):
        raise ValueError(f"RCS knots must be strictly increasing: {knots}")
    first, penultimate, last = knots[0], knots[-2], knots[-1]
    scale = (last - first) ** 2
    if scale <= 1e-12 or last - penultimate <= 1e-12:
        raise ValueError("Degenerate RCS knot range.")

    def cube_positive(values: np.ndarray, knot: float) -> np.ndarray:
        return np.maximum(values - knot, 0.0) ** 3

    columns = [x]
    for knot in knots[:-2]:
        nonlinear = (
            cube_positive(x, knot)
            - cube_positive(x, penultimate) * (last - knot) / (last - penultimate)
            + cube_positive(x, last) * (penultimate - knot) / (last - penultimate)
        ) / scale
        columns.append(nonlinear)
    return np.column_stack(columns)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--seed", type=int, default=42, choices=[42])
    result.add_argument("--folds", type=int, default=3, choices=[3])
    result.add_argument("--epochs", type=int, default=100, choices=[100])
    result.add_argument("--patience", type=int, default=5, choices=[5])
    result.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0"])
    return result


def core_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=0.0,
        inner_valid_fraction=0.15,
        batch_size=1024,
        encode_batch_size=4096,
        d_model=64,
        n_heads=4,
        n_layers=4,
        mlp_hidden_dim=128,
        latent_dim=32,
        dropout=0.10,
        lr=1e-3,
        weight_decay=1e-4,
        treatment_groups=4,
        outcome_weight=1.0,
        treatment_weight=0.30,
        sinkhorn_weight=0.10,
        mmd_weight=0.10,
        stability_weight=0.02,
        perturb_std=0.02,
        sinkhorn_epsilon=0.25,
        sinkhorn_iters=20,
        mmd_sigmas=[0.5, 1.0, 2.0, 4.0],
        device=args.device,
        save_oof_representations=False,
    )


def build_dose_response_figure(
    curve: pd.DataFrame,
    treatment: np.ndarray,
    nonlinear_p: float,
    nobs: int,
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    navy = "#2B5C85"
    band = "#A9C9DF"
    neutral = "#6E6E6E"
    hist_color = "#C9CED3"

    fig = plt.figure(figsize=(3.54, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.1, 1.0], hspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    ax_dist = fig.add_subplot(grid[1, 0], sharex=ax)

    ax.fill_between(
        curve["treatment"].to_numpy(),
        curve["ci_low"].to_numpy(),
        curve["ci_high"].to_numpy(),
        color=band,
        alpha=0.55,
        linewidth=0,
        label="Pointwise 95% CI",
    )
    ax.plot(
        curve["treatment"],
        curve["relative_effect"],
        color=navy,
        linewidth=1.8,
        label="Orthogonal RCS",
    )
    ax.plot(
        curve["treatment"],
        curve["linear_comparator"],
        color=neutral,
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="Linear comparator",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.65, alpha=0.7)
    ax.axvline(
        float(curve["baseline_treatment"].iloc[0]),
        color="#8A8A8A",
        linewidth=0.7,
        linestyle=":",
    )
    ax.set_ylabel("Outcome change relative\nto median dose")
    ax.set_title("Orthogonal dose–response sensitivity", loc="left", pad=5, fontweight="bold")
    ax.text(
        0.99,
        0.97,
        f"Joint nonlinearity $p$ = {nonlinear_p:.3f}\n$n$ = {nobs:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#333333",
    )
    ax.legend(loc="lower right", handlelength=2.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", color="#E8EAEC", linewidth=0.55)
    ax.set_axisbelow(True)

    lower = float(curve["treatment"].min())
    upper = float(curve["treatment"].max())
    visible = treatment[(treatment >= lower) & (treatment <= upper)]
    ax_dist.hist(visible, bins=32, color=hist_color, edgecolor="white", linewidth=0.25)
    ax_dist.axvline(
        float(curve["baseline_treatment"].iloc[0]),
        color="#8A8A8A",
        linewidth=0.7,
        linestyle=":",
    )
    ax_dist.set_ylabel("Count")
    ax_dist.set_xlabel("Search-attention dose, log(1 + SVI), winsorized")
    ax_dist.set_xlim(lower, upper)
    ax_dist.tick_params(axis="y", labelleft=False, length=0)
    ax_dist.spines["left"].set_visible(False)
    ax_dist.text(
        0.01,
        0.88,
        "Displayed support: 5th–95th percentile",
        transform=ax_dist.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#555555",
    )

    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.14)
    stem = output_dir / "figure_orthogonal_rcs_dose_response"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def output_manifest(output_dir: Path) -> None:
    excluded = {"OUTPUT_MANIFEST_SHA256.csv", "OUTPUT_MANIFEST_SHA256.csv.sha256"}
    rows = []
    for path in sorted(item for item in output_dir.glob("*") if item.is_file()):
        if path.name in excluded:
            continue
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = output_dir / "OUTPUT_MANIFEST_SHA256.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    (output_dir / "OUTPUT_MANIFEST_SHA256.csv.sha256").write_text(
        f"{sha256_file(manifest)}  OUTPUT_MANIFEST_SHA256.csv\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    core = load_core()
    model_args = core_args(args)
    config = Config()
    config.seed = args.seed
    set_seed(args.seed)
    bundle = prepare_dataset(config)
    X = np.asarray(bundle["x_all"], dtype=np.float32)
    Y = np.asarray(bundle["y_all"], dtype=np.float64)
    T = np.asarray(bundle["t_all"], dtype=np.float64)
    ordered_df = pd.concat(
        [bundle["train_df"], bundle["valid_df"], bundle["test_df"]],
        ignore_index=True,
    )
    if len(ordered_df) != len(Y):
        raise RuntimeError("Panel row metadata is not aligned with the prepared arrays.")
    years = ordered_df["year"].to_numpy(dtype=np.int64)
    firms = ordered_df["stkcd"].astype(str).to_numpy()

    calibration = T[years <= config.train_end]
    center = float(np.median(calibration))
    q25, q75 = np.quantile(calibration, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-8:
        raise RuntimeError("Calibration treatment IQR is degenerate.")
    T_std = (T - center) / scale
    calibration_std = (calibration - center) / scale
    knot_probabilities = np.array([0.05, 0.35, 0.65, 0.95], dtype=np.float64)
    knots_std = np.quantile(calibration_std, knot_probabilities)
    basis = restricted_cubic_spline_basis(T_std, knots_std)
    basis_names = ["rcs_linear_std", "rcs_nonlinear_1", "rcs_nonlinear_2"]
    if basis.shape[1] != len(basis_names):
        raise RuntimeError(f"Unexpected basis dimension: {basis.shape}")

    run_config = {
        "script_version": SCRIPT_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "folds": args.folds,
        "model": "Fully cross-fitted FT-Sinkhorn",
        "core_hyperparameters": vars(model_args),
        "rcs": {
            "basis": "restricted cubic spline: linear + 2 nonlinear columns",
            "knot_probabilities": knot_probabilities.tolist(),
            "knots_standardized": knots_std.tolist(),
            "knots_treatment_units": (center + scale * knots_std).tolist(),
            "calibration_period": f"year <= {config.train_end}",
            "calibration_center_median": center,
            "calibration_scale_iqr": scale,
            "curve_display_percentiles": [0.05, 0.95],
            "curve_baseline": "full-sample treatment median",
        },
        "orthogonalization": {
            "outcome_nuisance": "fold-specific LassoCV(Z_train -> Y_train)",
            "basis_nuisance": "one fold-specific LassoCV per RCS basis column",
            "final_regression": "Y residual on three RCS basis residuals",
            "covariance": "HC3",
        },
        "input_hashes": {
            "core_script_sha256": sha256_file(CORE_SCRIPT),
            "linear_effect_sha256": sha256_file(CORE_LINEAR_EFFECT_PATH),
            "search_data_sha256": sha256_file(config.search_path),
            "innovation_data_sha256": sha256_file(config.innovation_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    print(f"[VERSION] {SCRIPT_VERSION}", flush=True)
    print(
        f"[DATA] n={len(Y)}, x_dim={X.shape[1]}, years={years.min()}-{years.max()}, "
        f"firms={pd.Series(firms).nunique()}",
        flush=True,
    )
    print(
        f"[RCS] center={center:.6f}, scale_iqr={scale:.6f}, "
        f"knots={np.array2string(center + scale * knots_std, precision=6)}",
        flush=True,
    )

    y_hat = np.full(len(Y), np.nan, dtype=np.float64)
    basis_hat = np.full_like(basis, np.nan, dtype=np.float64)
    outer_fold = np.zeros(len(Y), dtype=np.int64)
    histories: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    spec = core.SPEC_LIBRARY["ft_sinkhorn"]
    device = core.resolve_device(args.device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            overlap = int(np.intersect1d(train_idx, test_idx).size)
            if overlap != 0:
                raise RuntimeError(f"Row overlap in fold {fold}")
            model, history, encoder_meta = core.fit_encoder_on_outer_training_fold(
                X_train=X[train_idx],
                Y_train=Y[train_idx],
                T_train=T[train_idx],
                spec=spec,
                args=model_args,
                seed=args.seed + 10_000 * fold,
                outer_fold=fold,
            )
            histories.append(history)
            z_train = core.encode_model(
                model,
                X[train_idx],
                batch_size=model_args.encode_batch_size,
                device=device,
            ).astype(np.float64)
            z_test = core.encode_model(
                model,
                X[test_idx],
                batch_size=model_args.encode_batch_size,
                device=device,
            ).astype(np.float64)

            y_model = LassoCV(
                cv=3,
                random_state=args.seed + fold,
                max_iter=10000,
                n_jobs=1,
            )
            y_model.fit(z_train, Y[train_idx])
            y_hat[test_idx] = y_model.predict(z_test)

            basis_alphas: list[float] = []
            for column in range(basis.shape[1]):
                basis_model = LassoCV(
                    cv=3,
                    random_state=args.seed + fold + 1000 * (column + 1),
                    max_iter=10000,
                    n_jobs=1,
                )
                basis_model.fit(z_train, basis[train_idx, column])
                basis_hat[test_idx, column] = basis_model.predict(z_test)
                basis_alphas.append(float(basis_model.alpha_))

            outer_fold[test_idx] = fold
            audits.append(
                {
                    "outer_fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_test_row_overlap_n": overlap,
                    "heldout_DY_used_for_encoder": False,
                    "heldout_basis_used_for_nuisance_training": False,
                    "best_epoch": int(encoder_meta["best_epoch"]),
                    "epochs_ran": int(encoder_meta["epochs_ran"]),
                    "y_alpha": float(y_model.alpha_),
                    "basis_alpha_linear": basis_alphas[0],
                    "basis_alpha_nonlinear_1": basis_alphas[1],
                    "basis_alpha_nonlinear_2": basis_alphas[2],
                }
            )
            print(
                f"[OOF fold={fold}] best_epoch={encoder_meta['best_epoch']}, "
                f"n_test={len(test_idx)}, y_alpha={y_model.alpha_:.6g}",
                flush=True,
            )
            del model, z_train, z_test
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not np.isfinite(y_hat).all() or not np.isfinite(basis_hat).all():
        raise RuntimeError("Incomplete/non-finite OOF nuisance predictions.")
    if np.any(outer_fold == 0):
        raise RuntimeError("Incomplete outer-fold assignment.")

    y_residual = Y - y_hat
    basis_residual = basis - basis_hat
    final_design = sm.add_constant(
        pd.DataFrame(basis_residual, columns=basis_names),
        has_constant="add",
    )
    fit = sm.OLS(y_residual, final_design).fit(cov_type="HC3")
    ci = fit.conf_int()
    coefficient_rows = []
    for name in fit.params.index:
        coefficient_rows.append(
            {
                "term": name,
                "estimate": float(fit.params[name]),
                "std_err_hc3": float(fit.bse[name]),
                "t_value": float(fit.tvalues[name]),
                "p_value": float(fit.pvalues[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }
        )
    coefficients = pd.DataFrame(coefficient_rows)

    restrictions = np.zeros((2, len(fit.params)), dtype=np.float64)
    restrictions[0, list(fit.params.index).index("rcs_nonlinear_1")] = 1.0
    restrictions[1, list(fit.params.index).index("rcs_nonlinear_2")] = 1.0
    nonlinear_test = fit.wald_test(restrictions, use_f=True, scalar=True)
    nonlinear_stat = float(np.asarray(nonlinear_test.statistic).reshape(-1)[0])
    nonlinear_p = float(np.asarray(nonlinear_test.pvalue).reshape(-1)[0])
    linearity_table = pd.DataFrame(
        [
            {
                "test": "Joint Wald: both nonlinear RCS terms equal zero",
                "statistic": nonlinear_stat,
                "df_num": 2,
                "df_denom": float(getattr(nonlinear_test, "df_denom", fit.df_resid)),
                "p_value": nonlinear_p,
                "covariance": "HC3",
                "nobs": int(fit.nobs),
                "decision_at_5pct": (
                    "reject_constant_marginal_effect" if nonlinear_p < 0.05 else "do_not_reject_linearity"
                ),
            }
        ]
    )

    display_low, display_high = np.quantile(T, [0.05, 0.95])
    baseline_treatment = float(np.median(T))
    grid = np.linspace(display_low, display_high, 240)
    grid_basis = restricted_cubic_spline_basis((grid - center) / scale, knots_std)
    baseline_basis = restricted_cubic_spline_basis(
        np.array([(baseline_treatment - center) / scale]),
        knots_std,
    )[0]
    contrasts = grid_basis - baseline_basis
    theta = fit.params[basis_names].to_numpy(dtype=np.float64)
    theta_cov = fit.cov_params().loc[basis_names, basis_names].to_numpy(dtype=np.float64)
    relative_effect = contrasts @ theta
    variance = np.einsum("ij,jk,ik->i", contrasts, theta_cov, contrasts)
    curve_se = np.sqrt(np.maximum(variance, 0.0))

    linear_comparator = pd.read_csv(CORE_LINEAR_EFFECT_PATH).iloc[0]
    linear_comparator_estimate = float(linear_comparator["estimate"])
    curve = pd.DataFrame(
        {
            "treatment": grid,
            "treatment_percentile_approx": np.searchsorted(np.sort(T), grid, side="right") / len(T),
            "baseline_treatment": baseline_treatment,
            "relative_effect": relative_effect,
            "std_err_hc3": curve_se,
            "ci_low": relative_effect - 1.959963984540054 * curve_se,
            "ci_high": relative_effect + 1.959963984540054 * curve_se,
            "linear_comparator": linear_comparator_estimate * (grid - baseline_treatment),
        }
    )

    oof = pd.DataFrame(
        {
            "row_id": np.arange(len(Y), dtype=np.int64),
            "stkcd": firms,
            "year": years,
            "outer_fold": outer_fold,
            "outcome": Y,
            "treatment": T,
            "outcome_hat_oof": y_hat,
            "outcome_residual": y_residual,
        }
    )
    for column, name in enumerate(basis_names):
        oof[f"basis__{name}"] = basis[:, column]
        oof[f"basis_hat_oof__{name}"] = basis_hat[:, column]
        oof[f"basis_residual__{name}"] = basis_residual[:, column]

    pd.concat(histories, ignore_index=True).to_csv(
        OUTPUT_DIR / "encoder_training_history.csv",
        index=False,
    )
    pd.DataFrame(audits).to_csv(OUTPUT_DIR / "fold_audit.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "rcs_orthogonal_coefficients.csv", index=False)
    linearity_table.to_csv(OUTPUT_DIR / "joint_nonlinearity_test.csv", index=False)
    curve.to_csv(OUTPUT_DIR / "figure_source_data.csv", index=False)
    oof.to_csv(OUTPUT_DIR / "oof_orthogonal_rcs_data.csv", index=False)
    pd.DataFrame(
        {
            "knot_probability": knot_probabilities,
            "knot_standardized": knots_std,
            "knot_treatment_units": center + scale * knots_std,
        }
    ).to_csv(OUTPUT_DIR / "rcs_knot_definition.csv", index=False)

    build_dose_response_figure(curve, T, nonlinear_p, len(Y), OUTPUT_DIR)
    legend = f"""**Figure 4 | Orthogonal dose–response sensitivity for search attention.**
The solid blue curve reports the FT–Sinkhorn orthogonal restricted-cubic-spline estimate relative to the sample median dose; shading is the pointwise 95% HC3 confidence interval. The dashed line is the linear-effect comparator using seed 42. Four knots are at the 5th, 35th, 65th, and 95th percentiles of the pre-2022 calibration-period treatment distribution. The lower panel shows treatment support. Estimation uses all {len(Y):,} observations; visualization is restricted to the 5th–95th percentile to avoid tail extrapolation. The joint Wald test for the two nonlinear terms gives p = {nonlinear_p:.4f}.
"""
    (OUTPUT_DIR / "FIGURE_LEGEND.md").write_text(legend, encoding="utf-8")

    run_config["status"] = "complete"
    run_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_config["result"] = {
        "joint_nonlinearity_statistic": nonlinear_stat,
        "joint_nonlinearity_p_value": nonlinear_p,
        "decision_at_5pct": linearity_table.iloc[0]["decision_at_5pct"],
        "nobs": int(fit.nobs),
        "condition_number": float(fit.condition_number),
        "linear_comparator_estimate": linear_comparator_estimate,
        "display_treatment_low_p05": float(display_low),
        "display_treatment_high_p95": float(display_high),
        "baseline_treatment_median": baseline_treatment,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    write_json(
        OUTPUT_DIR / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "completed_at_utc": run_config["completed_at_utc"],
            "nobs": int(fit.nobs),
            "all_oof_predictions_finite": True,
            "max_train_test_row_overlap_n": int(
                pd.DataFrame(audits)["train_test_row_overlap_n"].max()
            ),
            "joint_nonlinearity_p_value": nonlinear_p,
        },
    )
    output_manifest(OUTPUT_DIR)
    print("[COEFFICIENTS]", flush=True)
    print(coefficients.to_string(index=False), flush=True)
    print("[NONLINEARITY TEST]", flush=True)
    print(linearity_table.to_string(index=False), flush=True)
    print(f"[COMPLETE] {OUTPUT_DIR}", flush=True)


def main() -> None:
    args = parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
