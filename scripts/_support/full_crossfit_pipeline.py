from __future__ import annotations

"""Shared full-crossfit analysis pipeline."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy._lib._util as scipy_util
import statsmodels.api as sm
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


if not hasattr(scipy_util, "_lazywhere"):

    def _lazywhere(cond, arrays, f, fillvalue=np.nan, f2=None):
        arrays = [np.asarray(arr) for arr in arrays]
        cond = np.asarray(cond, dtype=bool)
        out = np.full(cond.shape, fillvalue, dtype=np.result_type(*arrays))
        if np.any(cond):
            out[cond] = f(*[arr[cond] for arr in arrays])
        if f2 is not None:
            inv = ~cond
            if np.any(inv):
                out[inv] = f2(*[arr[inv] for arr in arrays])
        return out

    scipy_util._lazywhere = _lazywhere


from econml.dml import CausalForestDML


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exp02_empirical_core_ablation_full_crossfit as core
from src.causal_utils import subgroup_cate_summary, top_bottom_gap
from src.config import Config
from src.data_utils import load_panel, prepare_dataset, set_seed, winsorize


SCRIPT_VERSION = "FULL_CROSSFIT_PIPELINE_20260802"
DEFAULT_OUTPUT_NAME = "full_crossfit_diagnostics_and_placebo"
FT_SINKHORN_SPEC = core.SPEC_LIBRARY["ft_sinkhorn"]
PLAIN_SPEC = core.SPEC_LIBRARY["plain_ft"]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_from_bundle(bundle: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for key in ["train_df", "valid_df", "test_df"]:
        frame = bundle[key].copy()
        frame["sample_split"] = key.replace("_df", "")
        frame["source_index"] = frame.index
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)


def r2_from_predictions(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denom = float(np.sum((actual - actual.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denom)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def model_arg_payload(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "epochs",
        "patience",
        "min_delta",
        "inner_valid_fraction",
        "batch_size",
        "encode_batch_size",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_hidden_dim",
        "latent_dim",
        "dropout",
        "lr",
        "weight_decay",
        "treatment_groups",
        "outcome_weight",
        "treatment_weight",
        "sinkhorn_weight",
        "mmd_weight",
        "stability_weight",
        "perturb_std",
        "sinkhorn_epsilon",
        "sinkhorn_iters",
        "mmd_sigmas",
        "device",
    ]
    return {key: getattr(args, key) for key in keys}


def make_splits(
    split_kind: str,
    X: np.ndarray,
    ordered_df: pd.DataFrame,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if split_kind == "random":
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(X)]
    if split_kind == "firm-group":
        groups = ordered_df["stkcd"].to_numpy()
        splitter = GroupKFold(n_splits=folds)
        return [(tr, te) for tr, te in splitter.split(X, groups=groups)]
    if split_kind == "year-group":
        groups = ordered_df["year"].to_numpy()
        n_splits = min(folds, int(pd.Series(groups).nunique()))
        splitter = GroupKFold(n_splits=n_splits)
        return [(tr, te) for tr, te in splitter.split(X, groups=groups)]
    raise ValueError(f"Unknown split kind: {split_kind}")


def group_overlap_audit(
    split_kind: str,
    ordered_df: pd.DataFrame,
    tr: np.ndarray,
    te: np.ndarray,
) -> tuple[int, str]:
    if split_kind == "firm-group":
        train_groups = set(ordered_df.iloc[tr]["stkcd"].tolist())
        test_groups = set(ordered_df.iloc[te]["stkcd"].tolist())
    elif split_kind == "year-group":
        train_groups = set(ordered_df.iloc[tr]["year"].tolist())
        test_groups = set(ordered_df.iloc[te]["year"].tolist())
    else:
        return 0, "not_applicable"
    overlap = train_groups.intersection(test_groups)
    return len(overlap), ",".join(map(str, sorted(test_groups)))


def fit_residual_regression(
    Y: np.ndarray,
    T: np.ndarray,
    y_hat: np.ndarray,
    t_hat: np.ndarray,
    covariance: str = "HC3",
    cluster_groups: np.ndarray | None = None,
) -> dict[str, Any]:
    y_res = np.asarray(Y, dtype=np.float64) - np.asarray(y_hat, dtype=np.float64)
    t_res = np.asarray(T, dtype=np.float64) - np.asarray(t_hat, dtype=np.float64)
    design = sm.add_constant(pd.DataFrame({"t_res": t_res}), has_constant="add")
    model = sm.OLS(y_res, design)
    if covariance == "firm-clustered":
        if cluster_groups is None:
            raise ValueError("cluster_groups is required for firm-clustered covariance")
        fit = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_groups, "use_correction": True},
        )
    else:
        fit = model.fit(cov_type="HC3")
    ci = fit.conf_int().loc["t_res"]
    return {
        "estimate": float(fit.params["t_res"]),
        "std_err": float(fit.bse["t_res"]),
        "t_value": float(fit.tvalues["t_res"]),
        "p_value": float(fit.pvalues["t_res"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "nobs": int(fit.nobs),
    }


def run_raw_crossfit(
    bundle: dict[str, Any],
    ordered_df: pd.DataFrame,
    split_kind: str,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    X = np.asarray(bundle["x_all"], dtype=np.float64)
    Y = np.asarray(bundle["y_all"], dtype=np.float64)
    T = np.asarray(bundle["t_all"], dtype=np.float64)
    y_hat = np.zeros(len(Y), dtype=np.float64)
    t_hat = np.zeros(len(T), dtype=np.float64)
    fold_id = np.zeros(len(Y), dtype=np.int64)
    rows = []
    splits = make_splits(split_kind, X, ordered_df, folds, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (tr, te) in enumerate(splits, start=1):
            ym = LassoCV(cv=3, random_state=seed + fold, max_iter=10000, n_jobs=1)
            tm = LassoCV(
                cv=3,
                random_state=seed + fold + 100,
                max_iter=10000,
                n_jobs=1,
            )
            ym.fit(X[tr], Y[tr])
            tm.fit(X[tr], T[tr])
            y_hat[te] = ym.predict(X[te])
            t_hat[te] = tm.predict(X[te])
            fold_id[te] = fold
            group_overlap, heldout_groups = group_overlap_audit(
                split_kind, ordered_df, tr, te
            )
            rows.append(
                {
                    "outer_fold": fold,
                    "split": split_kind,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "train_test_overlap_n": int(np.intersect1d(tr, te).size),
                    "train_test_group_overlap_n": group_overlap,
                    "heldout_groups": heldout_groups,
                    "y_alpha": float(ym.alpha_),
                    "t_alpha": float(tm.alpha_),
                    "y_r2_test": r2_from_predictions(Y[te], y_hat[te]),
                    "t_r2_test": r2_from_predictions(T[te], t_hat[te]),
                }
            )
    if np.any(fold_id == 0):
        raise RuntimeError(f"Incomplete Raw-X cross-fitting for {split_kind}")
    effect = fit_residual_regression(Y, T, y_hat, t_hat, covariance="HC3")
    return {
        "effect": effect,
        "y_hat": y_hat,
        "t_hat": t_hat,
        "fold_id": fold_id,
        "audit": pd.DataFrame(rows),
    }


def fit_geometry(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
    test_scaled = scaler.transform(np.asarray(X_test, dtype=np.float64))
    n_comp = min(n_components, train_scaled.shape[1], max(1, len(train_scaled) - 1))
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def pairwise_smd(X: np.ndarray, groups: np.ndarray) -> float:
    values = []
    for g1, g2 in combinations(sorted(np.unique(groups)), 2):
        x1 = X[groups == g1]
        x2 = X[groups == g2]
        if len(x1) < 2 or len(x2) < 2:
            continue
        pooled = np.sqrt((x1.var(axis=0, ddof=1) + x2.var(axis=0, ddof=1)) / 2.0)
        pooled[pooled < 1e-12] = 1.0
        values.append(float(np.mean(np.abs((x1.mean(axis=0) - x2.mean(axis=0)) / pooled))))
    return float(np.mean(values)) if values else float("nan")


def pairwise_distribution_distances(
    X: np.ndarray,
    groups: np.ndarray,
    max_per_group: int,
    seed: int,
    sigmas: list[float],
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    selected: dict[int, np.ndarray] = {}
    for group in sorted(np.unique(groups)):
        idx = np.flatnonzero(groups == group)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        selected[int(group)] = idx
    mmd_values = []
    sinkhorn_values = []
    for g1, g2 in combinations(sorted(selected), 2):
        if len(selected[g1]) == 0 or len(selected[g2]) == 0:
            continue
        x1 = torch.tensor(X[selected[g1]], dtype=torch.float32)
        x2 = torch.tensor(X[selected[g2]], dtype=torch.float32)
        mmd_values.append(float(core.mmd_pair_loss(x1, x2, sigmas=sigmas).item()))
        sinkhorn_values.append(
            float(
                core.sinkhorn_pair_loss(
                    x1,
                    x2,
                    epsilon=0.25,
                    n_iters=20,
                ).item()
            )
        )
    return (
        float(np.mean(mmd_values)) if mmd_values else float("nan"),
        float(np.mean(sinkhorn_values)) if sinkhorn_values else float("nan"),
    )


def geometry_fold_diagnostics(
    space: str,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    T_train: np.ndarray,
    T_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_geom, test_geom = fit_geometry(
        X_train,
        X_test,
        n_components=args.pca_components,
        seed=args.seed + 1_000 * fold,
    )
    q4_cut = core.fit_quantile_cutoffs(T_train, 4)
    d10_cut = core.fit_quantile_cutoffs(T_train, 10)
    q4_train = core.apply_quantile_groups(T_train, q4_cut)
    q4_test = core.apply_quantile_groups(T_test, q4_cut)
    d10_train = core.apply_quantile_groups(T_train, d10_cut)
    d10_test = core.apply_quantile_groups(T_test, d10_cut)

    k = min(args.nn_k, len(train_geom))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_geom)
    neighbor_index = nn.kneighbors(test_geom, return_distance=False)
    neighbor_q4 = q4_train[neighbor_index]
    neighbor_d10 = d10_train[neighbor_index]
    q4_cross = (neighbor_q4 != q4_test[:, None]).mean(axis=1)
    q4_same = 1.0 - q4_cross
    middle80 = ((neighbor_d10 >= 1) & (neighbor_d10 <= 8)).mean(axis=1)
    decile_same = (neighbor_d10 == d10_test[:, None]).mean(axis=1)

    scores = pd.DataFrame(
        {
            "space": space,
            "outer_fold": fold,
            "row_index": test_index,
            "treatment_q4": q4_test,
            "treatment_decile": d10_test,
            "q4_cross_share": q4_cross,
            "q4_same_share": q4_same,
            "middle80_neighbor_share": middle80,
            "decile_same_share": decile_same,
        }
    )
    mmd, sinkhorn = pairwise_distribution_distances(
        test_geom,
        q4_test,
        max_per_group=args.distance_sample_per_group,
        seed=args.seed + 10_000 * fold + sum(map(ord, space)),
        sigmas=args.mmd_sigmas,
    )
    summary = {
        "space": space,
        "outer_fold": fold,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "smd_pair_mean": pairwise_smd(test_geom, q4_test),
        "mmd_pair_mean": mmd,
        "sinkhorn_pair_mean": sinkhorn,
        "cross_group_neighbor_share_mean": float(q4_cross.mean()),
        "low_overlap_share_cross_lt_0_25": float((q4_cross < 0.25).mean()),
        "severe_low_overlap_share_cross_lt_0_10": float((q4_cross < 0.10).mean()),
    }
    return summary, scores


def fit_causal_forest_quiet(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    train_T: np.ndarray,
    test_X: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_X = np.asarray(train_X, dtype=np.float64)
    train_Y = np.asarray(train_Y, dtype=np.float64)
    train_T = np.asarray(train_T, dtype=np.float64)
    test_X = np.asarray(test_X, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        ym = LassoCV(cv=3, random_state=seed, max_iter=10000, n_jobs=1)
        tm = LassoCV(cv=3, random_state=seed + 1, max_iter=10000, n_jobs=1)
        forest = CausalForestDML(
            model_y=ym,
            model_t=tm,
            discrete_treatment=False,
            n_estimators=args.cf_n_estimators,
            min_samples_leaf=args.cf_min_samples_leaf,
            max_depth=args.cf_max_depth,
            honest=True,
            inference=True,
            cv=args.folds,
            random_state=seed,
            n_jobs=1,
        )
        forest.fit(Y=train_Y, T=train_T, X=train_X, W=None)
    cate = np.asarray(forest.effect(test_X), dtype=np.float64)
    low, high = forest.effect_interval(test_X)
    return cate, np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def run_fingerprint(
    run_id: str,
    bundle: dict[str, Any],
    spec: dict[str, str],
    split_kind: str,
    args: argparse.Namespace,
    geometry: bool,
    heterogeneity: bool,
    include_raw_geometry: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "script_version": SCRIPT_VERSION,
        "wrapper_source_hash": file_sha256(Path(__file__).resolve()),
        "core_source_hash": file_sha256(Path(core.__file__).resolve()),
        "run_id": run_id,
        "spec": spec,
        "split_kind": split_kind,
        "seed": args.seed,
        "folds": args.folds,
        "model_args": model_arg_payload(args),
        "geometry": geometry,
        "heterogeneity": heterogeneity,
        "include_raw_geometry": include_raw_geometry,
        "data_hash": array_sha256(
            np.asarray(bundle["x_all"]),
            np.asarray(bundle["y_all"]),
            np.asarray(bundle["t_all"]),
        ),
    }
    return payload_sha256(payload), payload


def load_completed_run(run_dir: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    complete_path = run_dir / "complete.json"
    if not complete_path.exists():
        return None
    metadata = json.loads(complete_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Checkpoint fingerprint mismatch in {run_dir}. "
            "Use a new output name rather than mixing specifications."
        )
    arrays = np.load(run_dir / "oof_arrays.npz")
    result: dict[str, Any] = {
        "effect": pd.read_csv(run_dir / "effect.csv").iloc[0].to_dict(),
        "y_hat": arrays["y_hat"],
        "t_hat": arrays["t_hat"],
        "z_oof": arrays["z_oof"],
        "fold_id": arrays["fold_id"],
        "audit": pd.read_csv(run_dir / "fold_audit.csv"),
        "history": pd.read_csv(run_dir / "training_history.csv"),
    }
    cate_path = run_dir / "cate_oof.npz"
    if cate_path.exists():
        cate = np.load(cate_path)
        result.update({key: cate[key] for key in cate.files})
    geometry_path = run_dir / "geometry_fold_summary.csv"
    scores_path = run_dir / "geometry_oof_scores.csv"
    result["geometry_summary"] = (
        pd.read_csv(geometry_path) if geometry_path.exists() else pd.DataFrame()
    )
    result["geometry_scores"] = (
        pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    )
    print(f"[RESUME] {run_dir.name}", flush=True)
    return result


def run_representation_crossfit(
    run_id: str,
    bundle: dict[str, Any],
    ordered_df: pd.DataFrame,
    spec: dict[str, str],
    split_kind: str,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    geometry: bool = False,
    heterogeneity: bool = False,
    include_raw_geometry: bool = False,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / run_id
    fingerprint, fingerprint_payload = run_fingerprint(
        run_id,
        bundle,
        spec,
        split_kind,
        args,
        geometry,
        heterogeneity,
        include_raw_geometry,
    )
    if args.resume:
        completed = load_completed_run(run_dir, fingerprint)
        if completed is not None:
            return completed
    elif (run_dir / "complete.json").exists():
        raise FileExistsError(
            f"Completed output already exists: {run_dir}. Use --resume or a new --output-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(bundle["x_all"], dtype=np.float32)
    Y = np.asarray(bundle["y_all"], dtype=np.float64)
    T = np.asarray(bundle["t_all"], dtype=np.float64)
    n = len(Y)
    y_hat = np.zeros(n, dtype=np.float64)
    t_hat = np.zeros(n, dtype=np.float64)
    z_oof = np.full((n, args.latent_dim), np.nan, dtype=np.float32)
    fold_id = np.zeros(n, dtype=np.int64)
    cate_model = np.full(n, np.nan, dtype=np.float64)
    cate_model_low = np.full(n, np.nan, dtype=np.float64)
    cate_model_high = np.full(n, np.nan, dtype=np.float64)
    cate_raw = np.full(n, np.nan, dtype=np.float64)
    cate_raw_low = np.full(n, np.nan, dtype=np.float64)
    cate_raw_high = np.full(n, np.nan, dtype=np.float64)
    histories = []
    audits = []
    geometry_rows = []
    geometry_scores = []
    splits = make_splits(split_kind, X, ordered_df, args.folds, args.seed)
    device = core.resolve_device(args.device)

    print(
        f"[RUN] {run_id}: spec={spec['spec_id']}, split={split_kind}, "
        f"n={n}, folds={len(splits)}",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (tr, te) in enumerate(splits, start=1):
            if np.intersect1d(tr, te).size:
                raise RuntimeError(f"Train/test overlap in {run_id}, fold {fold}")
            model, history, encoder_meta = core.fit_encoder_on_outer_training_fold(
                X_train=X[tr],
                Y_train=Y[tr],
                T_train=T[tr],
                spec=spec,
                args=args,
                seed=args.seed + 10_000 * fold,
                outer_fold=fold,
            )
            histories.append(history)
            z_train = core.encode_model(model, X[tr], args.encode_batch_size, device)
            z_test = core.encode_model(model, X[te], args.encode_batch_size, device)
            z_oof[te] = z_test
            fold_id[te] = fold

            ym = LassoCV(
                cv=3,
                random_state=args.seed + fold,
                max_iter=10000,
                n_jobs=1,
            )
            tm = LassoCV(
                cv=3,
                random_state=args.seed + fold + 100,
                max_iter=10000,
                n_jobs=1,
            )
            ym.fit(z_train.astype(np.float64), Y[tr])
            tm.fit(z_train.astype(np.float64), T[tr])
            y_hat[te] = ym.predict(z_test.astype(np.float64))
            t_hat[te] = tm.predict(z_test.astype(np.float64))

            group_overlap, heldout_groups = group_overlap_audit(
                split_kind, ordered_df, tr, te
            )
            audits.append(
                {
                    "run_id": run_id,
                    "spec_id": spec["spec_id"],
                    "outer_fold": fold,
                    "split": split_kind,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "train_test_overlap_n": int(np.intersect1d(tr, te).size),
                    "train_test_group_overlap_n": group_overlap,
                    "heldout_groups": heldout_groups,
                    "heldout_DY_used_for_encoder": False,
                    "best_epoch": encoder_meta["best_epoch"],
                    "epochs_ran": encoder_meta["epochs_ran"],
                    "best_valid_score": encoder_meta["best_valid_score"],
                    "y_alpha": float(ym.alpha_),
                    "t_alpha": float(tm.alpha_),
                }
            )

            if geometry:
                model_space = (
                    "BROL-FT-Sinkhorn (fold-specific)"
                    if spec["spec_id"] == "ft_sinkhorn"
                    else "Plain FT-Z (fold-specific)"
                )
                summary, scores = geometry_fold_diagnostics(
                    model_space,
                    fold,
                    tr,
                    te,
                    z_train,
                    z_test,
                    T[tr],
                    T[te],
                    args,
                )
                geometry_rows.append(summary)
                geometry_scores.append(scores)
                if include_raw_geometry:
                    raw_summary, raw_scores = geometry_fold_diagnostics(
                        "Raw-X",
                        fold,
                        tr,
                        te,
                        X[tr],
                        X[te],
                        T[tr],
                        T[te],
                        args,
                    )
                    geometry_rows.append(raw_summary)
                    geometry_scores.append(raw_scores)

            if heterogeneity:
                z_cate, z_low, z_high = fit_causal_forest_quiet(
                    z_train,
                    Y[tr],
                    T[tr],
                    z_test,
                    args,
                    seed=args.seed + 20_000 * fold,
                )
                raw_cate, raw_low, raw_high = fit_causal_forest_quiet(
                    X[tr],
                    Y[tr],
                    T[tr],
                    X[te],
                    args,
                    seed=args.seed + 20_000 * fold + 1,
                )
                cate_model[te], cate_model_low[te], cate_model_high[te] = (
                    z_cate,
                    z_low,
                    z_high,
                )
                cate_raw[te], cate_raw_low[te], cate_raw_high[te] = (
                    raw_cate,
                    raw_low,
                    raw_high,
                )

            del model, z_train, z_test
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if np.isnan(z_oof).any() or np.any(fold_id == 0):
        raise RuntimeError(f"Incomplete OOF representation for {run_id}")
    if heterogeneity and (np.isnan(cate_model).any() or np.isnan(cate_raw).any()):
        raise RuntimeError(f"Incomplete OOF heterogeneity predictions for {run_id}")

    effect = core.residual_hc3(Y, T, y_hat, t_hat)
    effect_row = core.summarize_effect(spec["name"], effect, spec["spec_id"])
    audit_df = pd.DataFrame(audits)
    history_df = pd.concat(histories, ignore_index=True)
    geometry_df = pd.DataFrame(geometry_rows)
    scores_df = (
        pd.concat(geometry_scores, ignore_index=True)
        if geometry_scores
        else pd.DataFrame()
    )

    pd.DataFrame([effect_row]).to_csv(run_dir / "effect.csv", index=False)
    audit_df.to_csv(run_dir / "fold_audit.csv", index=False)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    np.savez_compressed(
        run_dir / "oof_arrays.npz",
        y_hat=y_hat,
        t_hat=t_hat,
        z_oof=z_oof,
        fold_id=fold_id,
    )
    if heterogeneity:
        np.savez_compressed(
            run_dir / "cate_oof.npz",
            cate_model=cate_model,
            cate_model_low=cate_model_low,
            cate_model_high=cate_model_high,
            cate_raw=cate_raw,
            cate_raw_low=cate_raw_low,
            cate_raw_high=cate_raw_high,
        )
    if not geometry_df.empty:
        geometry_df.to_csv(run_dir / "geometry_fold_summary.csv", index=False)
        scores_df.to_csv(run_dir / "geometry_oof_scores.csv", index=False)
    write_json(
        run_dir / "complete.json",
        {
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "effect": effect_row,
            "completed": True,
        },
    )
    return {
        "effect": effect_row,
        "y_hat": y_hat,
        "t_hat": t_hat,
        "z_oof": z_oof,
        "fold_id": fold_id,
        "audit": audit_df,
        "history": history_df,
        "geometry_summary": geometry_df,
        "geometry_scores": scores_df,
        "cate_model": cate_model,
        "cate_model_low": cate_model_low,
        "cate_model_high": cate_model_high,
        "cate_raw": cate_raw,
        "cate_raw_low": cate_raw_low,
        "cate_raw_high": cate_raw_high,
    }


def weighted_geometry_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "smd_pair_mean",
        "mmd_pair_mean",
        "sinkhorn_pair_mean",
        "cross_group_neighbor_share_mean",
        "low_overlap_share_cross_lt_0_25",
        "severe_low_overlap_share_cross_lt_0_10",
    ]
    rows = []
    for space, group in frame.groupby("space", sort=False):
        weights = group["n_test"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "space": space,
            "nobs": int(group["n_test"].sum()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(values)
            row[metric] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any()
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        idx = rng.integers(0, len(values), size=len(values))
        means[rep] = values[idx].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_cate(
    space: str,
    cate: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    gap = top_bottom_gap(cate, reps=args.bootstrap_reps, seed=args.seed + seed_offset)
    ci_low, ci_high = bootstrap_mean_ci(
        cate, args.bootstrap_reps, args.seed + seed_offset + 1_000
    )
    return {
        "space": space,
        "forest_ate_hat": float(np.mean(cate)),
        "ate_ci_low_boot": ci_low,
        "ate_ci_high_boot": ci_high,
        "cate_sd": float(np.std(cate, ddof=1)),
        "cate_p05": float(np.quantile(cate, 0.05)),
        "cate_median": float(np.median(cate)),
        "cate_p95": float(np.quantile(cate, 0.95)),
        "top_bottom_gap": gap["gap"],
        "gap_se_boot": gap["se_boot"],
        "gap_p_value": gap["p_value"],
        "gap_ci_low": gap["ci_low"],
        "gap_ci_high": gap["ci_high"],
    }


def compare_cate(raw_cate: np.ndarray, model_cate: np.ndarray) -> pd.DataFrame:
    pear = pearsonr(raw_cate, model_cate)
    spear = spearmanr(raw_cate, model_cate)
    raw_q = np.asarray(pd.qcut(raw_cate, 10, labels=False, duplicates="drop"))
    model_q = np.asarray(pd.qcut(model_cate, 10, labels=False, duplicates="drop"))
    return pd.DataFrame(
        [
            {"metric": "pearson_corr", "value": float(pear.statistic), "p_value": float(pear.pvalue)},
            {"metric": "spearman_corr", "value": float(spear.statistic), "p_value": float(spear.pvalue)},
            {"metric": "same_sign_share", "value": float((np.sign(raw_cate) == np.sign(model_cate)).mean()), "p_value": np.nan},
            {"metric": "top_decile_overlap", "value": float(((raw_q == raw_q.max()) & (model_q == model_q.max())).sum() / max(1, (raw_q == raw_q.max()).sum())), "p_value": np.nan},
            {"metric": "bottom_decile_overlap", "value": float(((raw_q == raw_q.min()) & (model_q == model_q.min())).sum() / max(1, (raw_q == raw_q.min()).sum())), "p_value": np.nan},
        ]
    )


def build_overlap_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows4 = []
    for space, group in scores.groupby("space", sort=False):
        cross = group["q4_cross_share"]
        rows4.append(
            {
                "space": space,
                "mean_cross_quartile_neighbor_share": float(cross.mean()),
                "cross_quartile_share_ge_0_25": float((cross >= 0.25).mean()),
                "cross_quartile_share_ge_0_50": float((cross >= 0.50).mean()),
                "low_overlap_share_cross_lt_0_25": float((cross < 0.25).mean()),
                "severe_low_overlap_share_cross_lt_0_10": float((cross < 0.10).mean()),
                "nobs": int(len(group)),
            }
        )
    table4 = pd.DataFrame(rows4)

    rows5 = []
    for space, group in scores.groupby("space", sort=False):
        for q4, label in [(0, "Bottom quartile"), (3, "Top quartile")]:
            tail = group[group["treatment_q4"] == q4]
            cross = tail["q4_cross_share"]
            rows5.append(
                {
                    "space": space,
                    "tail": label,
                    "mean_nonown_quartile_neighbor_share": float(cross.mean()),
                    "common_support_share_cross_ge_0_20": float((cross >= 0.20).mean()),
                    "isolated_share_cross_lt_0_10": float((cross < 0.10).mean()),
                    "nobs": int(len(tail)),
                }
            )
    table5 = pd.DataFrame(rows5)

    rows6 = []
    for space, group in scores.groupby("space", sort=False):
        for decile, label in [(0, "Bottom decile"), (9, "Top decile")]:
            tail = group[group["treatment_decile"] == decile]
            middle = tail["middle80_neighbor_share"]
            rows6.append(
                {
                    "space": space,
                    "tail": label,
                    "middle80_neighbor_share": float(middle.mean()),
                    "middle80_support_share_ge_0_50": float((middle >= 0.50).mean()),
                    "isolated_tail_share_same_tail_gt_0_75": float((tail["decile_same_share"] > 0.75).mean()),
                    "nobs": int(len(tail)),
                }
            )
    return table4, table5, pd.DataFrame(rows6)


def stage_base(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    bundle = prepare_dataset(cfg)
    ordered_df = ordered_from_bundle(bundle)
    ft_sinkhorn = run_representation_crossfit(
        "base_ft_sinkhorn",
        bundle,
        ordered_df,
        FT_SINKHORN_SPEC,
        "random",
        output_dir,
        args,
        geometry=True,
        heterogeneity=True,
        include_raw_geometry=True,
    )
    plain = run_representation_crossfit(
        "base_plain_ft",
        bundle,
        ordered_df,
        PLAIN_SPEC,
        "random",
        output_dir,
        args,
        geometry=True,
        heterogeneity=False,
        include_raw_geometry=False,
    )
    raw = run_raw_crossfit(bundle, ordered_df, "random", args.folds, args.seed)
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base_dir / "raw_random_oof.npz",
        y_hat=raw["y_hat"],
        t_hat=raw["t_hat"],
        fold_id=raw["fold_id"],
    )
    pd.DataFrame([raw["effect"]]).to_csv(base_dir / "raw_random_effect.csv", index=False)
    raw["audit"].to_csv(base_dir / "raw_random_fold_audit.csv", index=False)

    geometry = pd.concat(
        [ft_sinkhorn["geometry_summary"], plain["geometry_summary"]], ignore_index=True
    )
    geometry_summary = weighted_geometry_summary(geometry)
    info = {
        "Raw-X": (
            r2_from_predictions(bundle["y_all"], raw["y_hat"]),
            r2_from_predictions(bundle["t_all"], raw["t_hat"]),
            bundle["x_all"].shape[1],
        ),
        "Plain FT-Z (fold-specific)": (
            r2_from_predictions(bundle["y_all"], plain["y_hat"]),
            r2_from_predictions(bundle["t_all"], plain["t_hat"]),
            args.latent_dim,
        ),
        "BROL-FT-Sinkhorn (fold-specific)": (
            r2_from_predictions(bundle["y_all"], ft_sinkhorn["y_hat"]),
            r2_from_predictions(bundle["t_all"], ft_sinkhorn["t_hat"]),
            args.latent_dim,
        ),
    }
    table3 = geometry_summary.copy()
    table3["outcome_pred_r2_oof"] = table3["space"].map(lambda x: info[x][0])
    table3["treatment_pred_r2_oof"] = table3["space"].map(lambda x: info[x][1])
    table3["dim"] = table3["space"].map(lambda x: info[x][2])
    table3.to_csv(output_dir / "table3.csv", index=False)

    ft_sinkhorn_scores = ft_sinkhorn["geometry_scores"]
    table4, table5, table6 = build_overlap_tables(ft_sinkhorn_scores)
    table4.to_csv(output_dir / "table4.csv", index=False)
    table5.to_csv(output_dir / "table5.csv", index=False)
    table6.to_csv(output_dir / "table6.csv", index=False)

    ols = core.ols_ate(bundle["x_all"], bundle["y_all"], bundle["t_all"])
    table7 = pd.DataFrame(
        [
            {"model": "OLS", **ols},
            {"model": "Raw-X DML", **raw["effect"]},
            {"model": "BROL-FT-Sinkhorn DML", **ft_sinkhorn["effect"]},
        ]
    )
    table7.to_csv(output_dir / "table7.csv", index=False)

    table8 = pd.DataFrame(
        [
            summarize_cate("Raw-X", ft_sinkhorn["cate_raw"], args, 10),
            summarize_cate("BROL-FT-Sinkhorn", ft_sinkhorn["cate_model"], args, 20),
        ]
    )
    table8.to_csv(output_dir / "table8.csv", index=False)
    group_cols = ["SOE", "HighTech", "Size", "Lev", "TobinQ"]
    raw_sub = subgroup_cate_summary(
        ordered_df,
        ft_sinkhorn["cate_raw"],
        group_cols,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    raw_sub.insert(0, "space", "Raw-X")
    model_sub = subgroup_cate_summary(
        ordered_df,
        ft_sinkhorn["cate_model"],
        group_cols,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    model_sub.insert(0, "space", "BROL-FT-Sinkhorn")
    table9 = pd.concat([raw_sub, model_sub], ignore_index=True)
    table9.to_csv(output_dir / "table9.csv", index=False)
    compare_cate(ft_sinkhorn["cate_raw"], ft_sinkhorn["cate_model"]).to_csv(
        base_dir / "cate_comparison.csv", index=False
    )
    cate_frame = ordered_df[
        ["stkcd", "year"] + [c for c in group_cols if c in ordered_df.columns]
    ].copy()
    cate_frame["outer_fold"] = ft_sinkhorn["fold_id"]
    cate_frame["cate_raw_oof"] = ft_sinkhorn["cate_raw"]
    cate_frame["cate_brol_oof"] = ft_sinkhorn["cate_model"]
    cate_frame.to_csv(base_dir / "cate_oof.csv", index=False)

    reference_path = ROOT / "outputs" / "exp02_empirical_core_ablation_full_crossfit" / "effect_ft_sinkhorn.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path).iloc[0]
        comparison = pd.DataFrame(
            [
                {
                    "metric": "estimate",
                    "reference": reference["estimate"],
                    "rerun": ft_sinkhorn["effect"]["estimate"],
                    "absolute_diff": abs(float(reference["estimate"]) - float(ft_sinkhorn["effect"]["estimate"])),
                },
                {
                    "metric": "std_err",
                    "reference": reference["std_err"],
                    "rerun": ft_sinkhorn["effect"]["std_err"],
                    "absolute_diff": abs(float(reference["std_err"]) - float(ft_sinkhorn["effect"]["std_err"])),
                },
            ]
        )
        comparison.to_csv(base_dir / "reference_reproducibility_check.csv", index=False)


def iterative_demean(
    frame: pd.DataFrame,
    cols: list[str],
    groups: tuple[str, ...] = ("stkcd", "year"),
    max_iter: int = 100,
    tol: float = 1e-10,
) -> pd.DataFrame:
    residualized = frame[cols].astype(float).to_numpy(copy=True)
    group_codes = [pd.factorize(frame[group], sort=False)[0] for group in groups]
    for _ in range(max_iter):
        old = residualized.copy()
        for codes in group_codes:
            temp = pd.DataFrame(residualized)
            temp["__group"] = codes
            means = temp.groupby("__group").transform("mean").iloc[:, : residualized.shape[1]].to_numpy()
            residualized -= means
        if np.max(np.abs(residualized - old)) < tol:
            break
    return pd.DataFrame(residualized, columns=cols)


def ols_firm_year_fe(cfg: Config, ordered_df: pd.DataFrame) -> dict[str, Any]:
    x_cols = [cfg.treatment] + [c for c in cfg.controls if c in ordered_df.columns]
    cols = [cfg.outcome] + x_cols
    work = ordered_df[["stkcd", "year"] + cols].copy()
    for col in cols:
        work[col] = work[col].fillna(work[col].median())
    residualized = iterative_demean(work, cols)
    fit = sm.OLS(residualized[cfg.outcome], residualized[x_cols]).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["stkcd"].to_numpy(), "use_correction": True},
    )
    ci = fit.conf_int().loc[cfg.treatment]
    return {
        "estimate": float(fit.params[cfg.treatment]),
        "std_err": float(fit.bse[cfg.treatment]),
        "t_value": float(fit.tvalues[cfg.treatment]),
        "p_value": float(fit.pvalues[cfg.treatment]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "nobs": int(fit.nobs),
    }


def load_base_run(output_dir: Path) -> dict[str, Any]:
    run_dir = output_dir / "runs" / "base_ft_sinkhorn"
    if not (run_dir / "complete.json").exists():
        raise FileNotFoundError("Base stage is required before this stage")
    arrays = np.load(run_dir / "oof_arrays.npz")
    return {
        "effect": pd.read_csv(run_dir / "effect.csv").iloc[0].to_dict(),
        "y_hat": arrays["y_hat"],
        "t_hat": arrays["t_hat"],
        "fold_id": arrays["fold_id"],
    }


def stage_panel(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    bundle = prepare_dataset(cfg)
    ordered_df = ordered_from_bundle(bundle)
    base = load_base_run(output_dir)
    raw_random_file = output_dir / "base" / "raw_random_oof.npz"
    if not raw_random_file.exists():
        raise FileNotFoundError("Raw-X base stage output is missing")
    raw_random = np.load(raw_random_file)
    rows = [
        {
            "model": "OLS firm+year FE",
            "split": "none",
            "covariance": "firm-clustered",
            **ols_firm_year_fe(cfg, ordered_df),
        }
    ]
    for label, y_hat, t_hat in [
        ("Raw-X DML", raw_random["y_hat"], raw_random["t_hat"]),
        ("BROL-FT-Sinkhorn DML", base["y_hat"], base["t_hat"]),
    ]:
        rows.append(
            {
                "model": label,
                "split": "random",
                "covariance": "HC3",
                **fit_residual_regression(
                    bundle["y_all"], bundle["t_all"], y_hat, t_hat, "HC3"
                ),
            }
        )
        rows.append(
            {
                "model": label,
                "split": "random",
                "covariance": "firm-clustered",
                **fit_residual_regression(
                    bundle["y_all"],
                    bundle["t_all"],
                    y_hat,
                    t_hat,
                    "firm-clustered",
                    ordered_df["stkcd"].to_numpy(),
                ),
            }
        )

    panel_dir = output_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    audit_frames = []
    for split_kind in ["firm-group", "year-group"]:
        model = run_representation_crossfit(
            f"panel_{split_kind.replace('-', '_')}_ft_sinkhorn",
            bundle,
            ordered_df,
            FT_SINKHORN_SPEC,
            split_kind,
            output_dir,
            args,
        )
        raw = run_raw_crossfit(bundle, ordered_df, split_kind, args.folds, args.seed)
        rows.append(
            {
                "model": "Raw-X DML",
                "split": split_kind,
                "covariance": "HC3",
                **raw["effect"],
            }
        )
        rows.append(
            {
                "model": "BROL-FT-Sinkhorn DML",
                "split": split_kind,
                "covariance": "HC3",
                **model["effect"],
            }
        )
        raw_audit = raw["audit"].copy()
        raw_audit.insert(0, "model", "Raw-X DML")
        model_audit = model["audit"].copy()
        model_audit.insert(0, "model", "BROL-FT-Sinkhorn DML")
        audit_frames.extend([raw_audit, model_audit])
    pd.DataFrame(rows).to_csv(output_dir / "table10.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(
        panel_dir / "group_split_fold_audit.csv", index=False
    )


def stage_alternative(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    specs = [
        ("Outcome = PG", "Patent_Award1", cfg.treatment, "alt_outcome_pg"),
        ("Outcome = IE", "InnoEff1", cfg.treatment, "alt_outcome_ie"),
        ("Outcome = RDI", "RD1", cfg.treatment, "alt_outcome_rdi"),
        ("Treatment = ASA", cfg.outcome, "SVI_All_year", "alt_treatment_asa"),
    ]
    if args.smoke:
        specs = specs[:1]
    rows = []
    for label, outcome, treatment, run_id in specs:
        alt_cfg = replace(cfg, outcome=outcome, treatment=treatment)
        bundle = prepare_dataset(alt_cfg)
        ordered_df = ordered_from_bundle(bundle)
        result = run_representation_crossfit(
            run_id,
            bundle,
            ordered_df,
            FT_SINKHORN_SPEC,
            "random",
            output_dir,
            args,
        )
        rows.append(
            {
                "specification": label,
                "outcome": outcome,
                "treatment": treatment,
                "representation_retrained": True,
                **result["effect"],
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output_dir / "table11.csv", index=False)


def load_raw_merged_panel(cfg: Config) -> pd.DataFrame:
    search_df = pd.read_excel(cfg.search_path, sheet_name="panel")
    innovation_df = pd.read_excel(cfg.innovation_path, sheet_name="panel")
    merge_cols = ["stkcd", "year"]
    search_cols = merge_cols + [
        c for c in cfg.search_core + cfg.controls if c in search_df.columns
    ]
    innovation_cols = merge_cols + [
        c for c in cfg.innovation_candidates if c in innovation_df.columns
    ]
    return (
        search_df[search_cols]
        .merge(innovation_df[innovation_cols], on=merge_cols, how="inner")
        .drop_duplicates(subset=merge_cols)
    )


def panel_from_raw_merged(
    cfg: Config,
    raw_merged: pd.DataFrame,
    excluded_years: set[int],
) -> pd.DataFrame:
    df = raw_merged.loc[~raw_merged["year"].isin(excluded_years)].copy()
    if cfg.treatment.startswith("SVI_"):
        df[cfg.treatment] = np.log1p(df[cfg.treatment].clip(lower=0))
    for col in [
        c for c in cfg.controls + [cfg.treatment, cfg.outcome] if c in df.columns
    ]:
        df[col] = winsorize(df[col], cfg.winsor_lower, cfg.winsor_upper)
    return (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["stkcd", "year", cfg.outcome, cfg.treatment])
        .copy()
    )


def stage_year(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    raw_merged = load_raw_merged_panel(cfg)
    years = sorted(int(value) for value in raw_merged["year"].dropna().unique())
    year_specs: list[tuple[str, set[int]]] = [
        (f"Exclude {year}", {year}) for year in years
    ]
    year_specs.append(("Exclude 2020-2021", {2020, 2021}))
    if args.smoke:
        year_specs = [("Exclude 2024", {2024})]
    rows = []
    for label, excluded in year_specs:
        df = panel_from_raw_merged(cfg, raw_merged, excluded)
        bundle = prepare_dataset(cfg, df=df)
        ordered_df = ordered_from_bundle(bundle)
        token = "_".join(map(str, sorted(excluded)))
        result = run_representation_crossfit(
            f"year_exclude_{token}",
            bundle,
            ordered_df,
            FT_SINKHORN_SPEC,
            "random",
            output_dir,
            args,
        )
        rows.append(
            {
                "specification": label,
                "excluded_years": ",".join(map(str, sorted(excluded))),
                "year_removed_before_preprocessing": True,
                "representation_retrained": True,
                **result["effect"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "table12.csv", index=False)


def stage_placebo(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    base_df = load_panel(cfg)
    reps = 1 if args.smoke else args.placebo_reps
    rows = []
    for rep in range(1, reps + 1):
        permutation_seed = args.seed + 500_000 + rep
        rng = np.random.default_rng(permutation_seed)
        fake_df = base_df.copy()
        for year in sorted(fake_df["year"].unique()):
            idx = fake_df.index[fake_df["year"] == year]
            values = fake_df.loc[idx, cfg.treatment].to_numpy(copy=True)
            fake_df.loc[idx, cfg.treatment] = values[rng.permutation(len(values))]
        bundle = prepare_dataset(cfg, df=fake_df)
        ordered_df = ordered_from_bundle(bundle)
        result = run_representation_crossfit(
            f"placebo_{rep:04d}",
            bundle,
            ordered_df,
            FT_SINKHORN_SPEC,
            "random",
            output_dir,
            args,
        )
        rows.append(
            {
                "rep": rep,
                "permutation_seed": permutation_seed,
                "permuted_within_year": True,
                "representation_retrained": True,
                **result["effect"],
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / "placebo_draws.csv", index=False)
        print(f"[PLACEBO] completed {rep}/{reps}", flush=True)
    draws = pd.DataFrame(rows)
    actual = load_base_run(output_dir)["effect"]
    abs_exceed = int((draws["estimate"].abs() >= abs(float(actual["estimate"]))).sum())
    upper_exceed = int((draws["estimate"] >= float(actual["estimate"])).sum())
    summary = pd.DataFrame(
        [
            {
                "actual_ate": float(actual["estimate"]),
                "actual_std_err": float(actual["std_err"]),
                "actual_p_value": float(actual["p_value"]),
                "placebo_mean": float(draws["estimate"].mean()),
                "placebo_std": float(draws["estimate"].std(ddof=1)) if len(draws) > 1 else np.nan,
                "placebo_q01": float(draws["estimate"].quantile(0.01)),
                "placebo_q05": float(draws["estimate"].quantile(0.05)),
                "placebo_q50": float(draws["estimate"].quantile(0.50)),
                "placebo_q95": float(draws["estimate"].quantile(0.95)),
                "placebo_q99": float(draws["estimate"].quantile(0.99)),
                "abs_exceed_count": abs_exceed,
                "upper_exceed_count": upper_exceed,
                "empirical_p_abs_plus_one": (abs_exceed + 1) / (len(draws) + 1),
                "empirical_p_upper_plus_one": (upper_exceed + 1) / (len(draws) + 1),
                "reps": int(len(draws)),
                "full_pipeline_refit_each_draw": True,
            }
        ]
    )
    summary.to_csv(output_dir / "table13.csv", index=False)


def write_protocol(output_dir: Path, args: argparse.Namespace, stages: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(core.__file__).resolve(), snapshot / Path(core.__file__).name)
    shutil.copy2(Path(__file__).resolve(), snapshot / Path(__file__).name)
    write_json(
        output_dir / "run_protocol.json",
        {
            "script_version": SCRIPT_VERSION,
            "script_path": str(Path(__file__).resolve()),
            "core_model_source": str(Path(core.__file__).resolve()),
            "core_model_sha256": file_sha256(Path(core.__file__).resolve()),
            "model_rule": "BROL-FT-Sinkhorn from the shared full-crossfit script",
            "common_z_used": False,
            "fixed_z_used_for_inference": False,
            "heldout_DY_used_for_encoder": False,
            "stages": stages,
            "args": vars(args),
            "existing_output_directories_modified": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
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
    parser.add_argument("--mmd-sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--distance-sample-per-group", type=int, default=600)
    parser.add_argument("--nn-k", type=int, default=50)
    parser.add_argument("--cf-n-estimators", type=int, default=160)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=40)
    parser.add_argument("--cf-max-depth", type=int, default=10)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--placebo-reps", type=int, default=100)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", "base", "panel", "alternative", "year", "placebo"],
        default=["all"],
    )
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.folds = 2
        args.epochs = 2
        args.patience = 1
        args.cf_n_estimators = 40
        args.bootstrap_reps = 20
        args.distance_sample_per_group = min(args.distance_sample_per_group, 100)
    if args.cf_n_estimators % 4 != 0:
        raise ValueError("--cf-n-estimators must be divisible by 4")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    core.validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    stages = (
        ["base", "panel", "alternative", "year", "placebo"]
        if "all" in args.stages
        else list(dict.fromkeys(args.stages))
    )
    output_name = args.output_name.strip()
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("--output-name must be one folder name")
    output_dir = ROOT / "outputs" / output_name
    write_protocol(output_dir, args, stages)
    set_seed(args.seed)
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}", flush=True)
    print(f"[CORE MODEL SOURCE] {Path(core.__file__).resolve()}", flush=True)
    print(f"[OUTPUT] {output_dir}", flush=True)
    print(f"[STAGES] {', '.join(stages)}", flush=True)

    for stage in stages:
        print(f"\n===== STAGE {stage.upper()} =====", flush=True)
        if stage == "base":
            stage_base(output_dir, args)
        elif stage == "panel":
            stage_panel(output_dir, args)
        elif stage == "alternative":
            stage_alternative(output_dir, args)
        elif stage == "year":
            stage_year(output_dir, args)
        elif stage == "placebo":
            stage_placebo(output_dir, args)
        else:
            raise AssertionError(stage)

    write_json(
        output_dir / "run_status.json",
        {
            "script_version": SCRIPT_VERSION,
            "status": "completed_requested_stages",
            "completed_stages": stages,
            "output_dir": str(output_dir),
        },
    )
    print(f"\nCompleted requested stages. Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
