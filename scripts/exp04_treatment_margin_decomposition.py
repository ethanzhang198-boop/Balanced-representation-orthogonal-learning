from __future__ import annotations

"""Treatment-margin and temporal DML analyses."""

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

import _support.full_crossfit_pipeline as base
import exp02_empirical_core_ablation_full_crossfit as core
from src.config import Config
from src.data_utils import build_preprocessor, set_seed, split_panel, winsorize


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT_VERSION = "TREATMENT_MARGIN_TEMPORAL_DML_20260803"
DEFAULT_OUTPUT_NAME = "exp04_treatment_margin_decomposition"
FT_SINKHORN_SPEC = core.SPEC_LIBRARY["ft_sinkhorn"]
RAW_FSA = "SVI_code_year"
PA = "Patent1"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_panel(cfg: Config) -> pd.DataFrame:
    search = pd.read_excel(cfg.search_path, sheet_name="panel")
    innovation = pd.read_excel(cfg.innovation_path, sheet_name="panel")
    keys = ["stkcd", "year"]
    search_cols = keys + [c for c in cfg.search_core + cfg.controls if c in search.columns]
    innovation_cols = keys + [c for c in cfg.innovation_candidates if c in innovation.columns]
    frame = search[search_cols].merge(innovation[innovation_cols], on=keys, how="inner")
    frame = frame.drop_duplicates(subset=keys).copy()
    if frame.duplicated(keys).any():
        raise RuntimeError("Raw firm-year panel is not unique")
    if RAW_FSA not in frame or PA not in frame:
        raise KeyError(f"Required raw variables are missing: {RAW_FSA}, {PA}")
    frame["anchor_key"] = frame["stkcd"].astype(str) + "_" + frame["year"].astype(int).astype(str)
    return frame


def apply_analysis_preprocessing(frame: pd.DataFrame, cfg: Config, *, binary_treatment: bool) -> pd.DataFrame:
    """Apply the current pipeline's winsorization, without touching raw_fsa."""
    out = frame.copy()
    for col in [c for c in cfg.controls if c in out.columns] + ["outcome_value"]:
        out[col] = winsorize(out[col], cfg.winsor_lower, cfg.winsor_upper)
    if not binary_treatment:
        out["treatment_value"] = winsorize(out["treatment_value"], cfg.winsor_lower, cfg.winsor_upper)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["stkcd", "year", "outcome_value", "treatment_value"]).copy()
    if out.empty:
        raise RuntimeError("No observations remain after treatment/outcome eligibility and preprocessing")
    if binary_treatment and not set(out["treatment_value"].unique()).issubset({0.0, 1.0}):
        raise RuntimeError("Binary treatment was changed during preprocessing")
    return out


def prepare_dataset(frame: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Apply the configured controls, calendar-year preprocessing, and split."""
    feature_cols = [c for c in cfg.controls if c in frame.columns] + ["year"]
    numeric_cols = [c for c in cfg.controls if c in frame.columns]
    train_df, valid_df, test_df = split_panel(frame, cfg.train_end, cfg.valid_year)
    if train_df.empty:
        raise RuntimeError("Configured preprocessing training bucket is unexpectedly empty")
    preprocessor = build_preprocessor(feature_cols, numeric_cols, ["year"])
    preprocessor.fit(train_df[feature_cols])

    def transform(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if part.empty:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        x = preprocessor.transform(part[feature_cols])
        if hasattr(x, "toarray"):
            x = x.toarray()
        return (
            np.asarray(x, dtype=np.float32),
            part["outcome_value"].to_numpy(dtype=np.float32),
            part["treatment_value"].to_numpy(dtype=np.float32),
        )

    x_train, y_train, t_train = transform(train_df)
    x_valid, y_valid, t_valid = transform(valid_df)
    x_test, y_test, t_test = transform(test_df)
    if x_valid.shape[1] == 0:
        x_valid = np.empty((0, x_train.shape[1]), dtype=np.float32)
    if x_test.shape[1] == 0:
        x_test = np.empty((0, x_train.shape[1]), dtype=np.float32)
    return {
        "df": frame,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "x_all": np.vstack([x_train, x_valid, x_test]),
        "y_all": np.concatenate([y_train, y_valid, y_test]),
        "t_all": np.concatenate([t_train, t_valid, t_test]),
    }


def ordered_from_bundle(bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.concat([bundle["train_df"], bundle["valid_df"], bundle["test_df"]], ignore_index=True)


def margin_design(raw: pd.DataFrame, cfg: Config, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw.copy()
    original_zero = frame[RAW_FSA].le(0)
    if name == "baseline_continuous":
        frame["treatment_value"] = np.log1p(frame[RAW_FSA].clip(lower=0))
        binary = False
        estimand = "Full-sample partial-linear effect of a one-unit increase in log(1+FSA)."
    elif name == "extensive_margin":
        frame["treatment_value"] = np.where(
            frame[RAW_FSA].notna(), (frame[RAW_FSA] > 0).astype(float), np.nan
        )
        binary = True
        estimand = "Full-sample partial-linear contrast for any versus no raw FSA attention."
    elif name == "intensive_margin":
        frame = frame.loc[frame[RAW_FSA] > 0].copy()
        frame["treatment_value"] = np.log1p(frame[RAW_FSA])
        binary = False
        estimand = "Positive-FSA-subsample partial-linear effect of a one-unit increase in log(1+FSA); it differs from the full-sample estimand."
    else:
        raise ValueError(name)
    frame["outcome_value"] = frame[PA]
    frame = apply_analysis_preprocessing(frame, cfg, binary_treatment=binary)
    audit = {
        "specification": name,
        "analysis_family": "margin",
        "binary_treatment": binary,
        "estimand": estimand,
        "n_anchor_raw": int(len(raw)),
        "n_raw_zero_before_filter": int(original_zero.sum()),
        "n_after_positive_filter": int((raw[RAW_FSA] > 0).sum()) if name == "intensive_margin" else int(len(raw)),
        "n_analysis": int(len(frame)),
        "n_raw_zero_analysis": int((frame[RAW_FSA] <= 0).sum()),
        "raw_zero_share_analysis": float((frame[RAW_FSA] <= 0).mean()),
        "n_firms": int(frame["stkcd"].nunique()),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "dropped_missing_future": 0,
    }
    return frame, audit


def _future_join(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    lookup = raw[["stkcd", "year", RAW_FSA, PA]].copy()
    lookup["year"] = lookup["year"] - horizon
    return lookup.rename(columns={RAW_FSA: f"fsa_tplus{horizon}", PA: f"pa_tplus{horizon}"})


def temporal_design(raw: pd.DataFrame, cfg: Config, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = raw.copy()
    current["pa_t"] = current[PA]
    current["fsa_t"] = current[RAW_FSA]
    horizon = 0
    needs_future = False
    if name in {"fsa_t_to_pa_tplus1", "matched_pa_tplus1_baseline", "fsa_tplus1_to_pa_t"}:
        horizon = 1
        needs_future = True
    elif name in {"fsa_t_to_pa_tplus2", "matched_pa_tplus2_baseline"}:
        horizon = 2
        needs_future = True
    if needs_future:
        current = current.merge(_future_join(raw, horizon), on=["stkcd", "year"], how="left", validate="one_to_one")
    if name == "fsa_t_to_pa_t":
        current["treatment_value"] = np.log1p(current["fsa_t"].clip(lower=0))
        current["outcome_value"] = current["pa_t"]
        required_future = pd.Series(True, index=current.index)
        future_label = "none"
    elif name == "fsa_t_to_pa_tplus1":
        current["treatment_value"] = np.log1p(current["fsa_t"].clip(lower=0))
        current["outcome_value"] = current["pa_tplus1"]
        required_future = current["pa_tplus1"].notna() & current["pa_t"].notna()
        future_label = "PA_tplus1"
    elif name == "fsa_t_to_pa_tplus2":
        current["treatment_value"] = np.log1p(current["fsa_t"].clip(lower=0))
        current["outcome_value"] = current["pa_tplus2"]
        required_future = current["pa_tplus2"].notna() & current["pa_t"].notna()
        future_label = "PA_tplus2"
    elif name == "fsa_tplus1_to_pa_t":
        current["treatment_value"] = np.log1p(current["fsa_tplus1"].clip(lower=0))
        current["outcome_value"] = current["pa_t"]
        required_future = current["fsa_tplus1"].notna()
        future_label = "FSA_tplus1"
    elif name == "matched_pa_tplus1_baseline":
        current["treatment_value"] = np.log1p(current["fsa_t"].clip(lower=0))
        current["outcome_value"] = current["pa_t"]
        required_future = current["pa_tplus1"].notna() & current["pa_t"].notna()
        future_label = "matched PA_tplus1 availability"
    elif name == "matched_pa_tplus2_baseline":
        current["treatment_value"] = np.log1p(current["fsa_t"].clip(lower=0))
        current["outcome_value"] = current["pa_t"]
        required_future = current["pa_tplus2"].notna() & current["pa_t"].notna()
        future_label = "matched PA_tplus2 availability"
    else:
        raise ValueError(name)
    before_future = len(current)
    current = current.loc[required_future].copy()
    current[RAW_FSA] = current["fsa_t"]
    current = apply_analysis_preprocessing(current, cfg, binary_treatment=False)
    audit = {
        "specification": name,
        "analysis_family": "temporal",
        "binary_treatment": False,
        "future_requirement": future_label,
        "n_anchor_raw": int(before_future),
        "n_analysis": int(len(current)),
        "n_firms": int(current["stkcd"].nunique()),
        "year_min": int(current["year"].min()),
        "year_max": int(current["year"].max()),
        "dropped_missing_future": int(before_future - required_future.sum()),
        "n_raw_zero_analysis": int((current[RAW_FSA] <= 0).sum()),
        "raw_zero_share_analysis": float((current[RAW_FSA] <= 0).mean()),
    }
    return current, audit


def fit_encoder_fold(
    X_train: np.ndarray, Y_train: np.ndarray, T_train: np.ndarray, args: argparse.Namespace,
    seed: int, outer_fold: int, binary_treatment: bool,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    all_idx = np.arange(len(X_train))
    fit_idx, valid_idx = train_test_split(all_idx, test_size=args.inner_valid_fraction, shuffle=True, random_state=seed)
    x_fit, x_valid = np.asarray(X_train[fit_idx], np.float32), np.asarray(X_train[valid_idx], np.float32)
    y_fit, y_valid = np.asarray(Y_train[fit_idx], float), np.asarray(Y_train[valid_idx], float)
    t_fit, t_valid = np.asarray(T_train[fit_idx], float), np.asarray(T_train[valid_idx], float)
    y_fit_s, y_valid_s = core.standardize_from_train(y_fit, y_fit, y_valid)
    t_fit_s, t_valid_s = core.standardize_from_train(t_fit, t_fit, t_valid)
    if binary_treatment:
        if set(np.unique(t_fit)) != {0.0, 1.0}:
            raise RuntimeError(f"Binary treatment fit partition in fold {outer_fold} lacks a treatment class")
        fit_groups, valid_groups = t_fit.astype(np.int64), t_valid.astype(np.int64)
        grouping_rule = "exact_binary_0_1"
    else:
        cutoffs = core.fit_quantile_cutoffs(t_fit, args.treatment_groups)
        fit_groups, valid_groups = core.apply_quantile_groups(t_fit, cutoffs), core.apply_quantile_groups(t_valid, cutoffs)
        grouping_rule = f"{args.treatment_groups}_quantiles_fit_partition"
    device = core.resolve_device(args.device)
    set_seed(seed)
    model = core.make_model(FT_SINKHORN_SPEC, X_train.shape[1], args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xtr = torch.tensor(x_fit, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_fit_s.reshape(-1, 1), dtype=torch.float32, device=device)
    tin = torch.tensor(t_fit_s.reshape(-1, 1), dtype=torch.float32, device=device)
    ttarget = torch.tensor(t_fit.reshape(-1, 1), dtype=torch.float32, device=device)
    gtr = torch.tensor(fit_groups, dtype=torch.long, device=device)
    xva = torch.tensor(x_valid, dtype=torch.float32, device=device)
    yva = torch.tensor(y_valid_s.reshape(-1, 1), dtype=torch.float32, device=device)
    tvain = torch.tensor(t_valid_s.reshape(-1, 1), dtype=torch.float32, device=device)
    tvatarget = torch.tensor(t_valid.reshape(-1, 1), dtype=torch.float32, device=device)
    gva = torch.tensor(valid_groups, dtype=torch.long, device=device)
    best_state, best_score, wait = None, float("inf"), 0
    history: list[dict[str, Any]] = []
    n_batches = math.ceil(xtr.size(0) / args.batch_size)
    balance_weight = core.balance_weight_for_spec(FT_SINKHORN_SPEC, args)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {k: 0.0 for k in ["loss", "outcome", "treatment", "balance", "stability"]}
        order = torch.randperm(xtr.size(0), device=device)
        for batch in range(n_batches):
            idx = order[batch * args.batch_size:(batch + 1) * args.batch_size]
            bx, by, bti, btt, bg = xtr[idx], ytr[idx], tin[idx], ttarget[idx], gtr[idx]
            optimizer.zero_grad(set_to_none=True)
            z, d_hat, y_hat = model(bx, bti)
            outcome_loss = F.smooth_l1_loss(y_hat, by)
            treatment_loss = F.binary_cross_entropy_with_logits(d_hat, btt) if binary_treatment else F.smooth_l1_loss(d_hat, bti)
            balance_loss = core.compute_balance_loss(z, bg, FT_SINKHORN_SPEC["balance"], args)
            stability_loss = F.mse_loss(model.encode(bx + torch.randn_like(bx) * args.perturb_std), z.detach()) if args.stability_weight > 0 else z.new_tensor(0.0)
            loss = args.outcome_weight * outcome_loss + args.treatment_weight * treatment_loss + balance_weight * balance_loss + args.stability_weight * stability_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            n_batch = bx.size(0)
            for key, value in [("loss", loss), ("outcome", outcome_loss), ("treatment", treatment_loss), ("balance", balance_loss), ("stability", stability_loss)]:
                totals[key] += float(value.item()) * n_batch
        model.eval()
        with torch.no_grad():
            zva, dva, yva_hat = model(xva, tvain)
            valid_outcome = float(F.mse_loss(yva_hat, yva).item())
            valid_treatment = float((F.binary_cross_entropy_with_logits(dva, tvatarget) if binary_treatment else F.mse_loss(dva, tvain)).item())
            valid_balance = float(core.compute_balance_loss(zva, gva, FT_SINKHORN_SPEC["balance"], args).item())
        valid_score = args.outcome_weight * valid_outcome + args.treatment_weight * valid_treatment + balance_weight * valid_balance
        row = {"outer_fold": outer_fold, "epoch": epoch, "treatment_loss": "binary_cross_entropy_logits" if binary_treatment else "smooth_l1_standardized", "sinkhorn_grouping": grouping_rule, "train_loss": totals["loss"] / xtr.size(0), "train_outcome_loss": totals["outcome"] / xtr.size(0), "train_treatment_loss": totals["treatment"] / xtr.size(0), "train_balance_loss": totals["balance"] / xtr.size(0), "train_stability_loss": totals["stability"] / xtr.size(0), "valid_outcome_mse": valid_outcome, "valid_treatment_loss": valid_treatment, "valid_balance_loss": valid_balance, "valid_score": float(valid_score)}
        history.append(row)
        print(f"[BROL fold={outer_fold}] epoch={epoch:03d} train={row['train_loss']:.4f} val={valid_score:.4f}", flush=True)
        if valid_score < best_score - args.min_delta:
            best_score, best_state, wait = float(valid_score), deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Encoder did not select a valid state")
    model.load_state_dict(best_state); model.to(device)
    history_df = pd.DataFrame(history)
    best = history_df.loc[history_df["valid_score"].idxmin()]
    return model, history_df, {"best_epoch": int(best["epoch"]), "epochs_ran": int(len(history_df)), "best_valid_score": float(best_score), "treatment_loss": best["treatment_loss"], "sinkhorn_grouping": grouping_rule, "heldout_DY_used_for_encoder": False}


def run_ft_sinkhorn_dml(bundle: dict[str, Any], ordered: pd.DataFrame, args: argparse.Namespace, run_dir: Path, binary_treatment: bool) -> dict[str, Any]:
    X, Y, T = np.asarray(bundle["x_all"], np.float32), np.asarray(bundle["y_all"], float), np.asarray(bundle["t_all"], float)
    splits = base.make_splits("random", X, ordered, args.folds, args.seed)
    y_hat, t_hat, folds = np.zeros(len(Y)), np.zeros(len(T)), np.zeros(len(T), dtype=int)
    z_oof = np.full((len(Y), args.latent_dim), np.nan, dtype=np.float32)
    histories, audit = [], []
    device = core.resolve_device(args.device)
    for fold, (tr, te) in enumerate(splits, 1):
        if np.intersect1d(tr, te).size:
            raise RuntimeError("Outer fold overlap")
        model, history, meta = fit_encoder_fold(X[tr], Y[tr], T[tr], args, args.seed + 10000 * fold, fold, binary_treatment)
        z_train, z_test = core.encode_model(model, X[tr], args.encode_batch_size, device), core.encode_model(model, X[te], args.encode_batch_size, device)
        ym = LassoCV(cv=3, random_state=args.seed + fold, max_iter=10000, n_jobs=1)
        tm = LassoCV(cv=3, random_state=args.seed + fold + 100, max_iter=10000, n_jobs=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            ym.fit(z_train.astype(float), Y[tr]); tm.fit(z_train.astype(float), T[tr])
        y_hat[te], t_hat[te], z_oof[te], folds[te] = ym.predict(z_test.astype(float)), tm.predict(z_test.astype(float)), z_test, fold
        histories.append(history)
        audit.append({"outer_fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)), "train_test_overlap_n": int(np.intersect1d(tr, te).size), "heldout_DY_used_for_encoder": False, "y_alpha": float(ym.alpha_), "t_alpha": float(tm.alpha_), **meta})
        del model, z_train, z_test
        if device.type == "cuda": torch.cuda.empty_cache()
    if np.isnan(z_oof).any() or np.any(folds == 0): raise RuntimeError("Incomplete BROL OOF values")
    effect = core.residual_hc3(Y, T, y_hat, t_hat)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(histories, ignore_index=True).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(audit).to_csv(run_dir / "fold_audit.csv", index=False)
    pd.DataFrame({"outer_fold": folds, "y_hat_oof": y_hat, "t_hat_oof": t_hat}).to_csv(run_dir / "oof_nuisance.csv", index=False)
    np.savez_compressed(run_dir / "oof_arrays.npz", y_hat=y_hat, t_hat=t_hat, z_oof=z_oof, fold_id=folds)
    return {"effect": effect, "y_hat": y_hat, "t_hat": t_hat, "fold_id": folds, "audit": pd.DataFrame(audit)}


def run_raw(bundle: dict[str, Any], ordered: pd.DataFrame, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    result = base.run_raw_crossfit(bundle, ordered, "random", args.folds, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    result["audit"].to_csv(run_dir / "fold_audit.csv", index=False)
    pd.DataFrame({"outer_fold": result["fold_id"], "y_hat_oof": result["y_hat"], "t_hat_oof": result["t_hat"]}).to_csv(run_dir / "oof_nuisance.csv", index=False)
    return result


def effect_row(family: str, spec: str, model: str, result: dict[str, Any], Y: np.ndarray, T: np.ndarray, audit: dict[str, Any]) -> dict[str, Any]:
    eff = result["effect"]
    return {"analysis_family": family, "specification": spec, "model": model, "nobs": int(eff["nobs"]), "estimate": float(eff["estimate"]), "std_err": float(eff["std_err"]), "t_value": float(eff["t_value"]), "p_value": float(eff["p_value"]), "ci_low": float(eff["ci_low"]), "ci_high": float(eff["ci_high"]), "outcome_oof_r2": base.r2_from_predictions(Y, result["y_hat"]), "treatment_oof_r2": base.r2_from_predictions(T, result["t_hat"]), "binary_treatment": bool(audit["binary_treatment"]), "n_firms": int(audit["n_firms"]), "year_min": int(audit["year_min"]), "year_max": int(audit["year_max"]), "n_raw_zero_analysis": int(audit["n_raw_zero_analysis"]), "raw_zero_share_analysis": float(audit["raw_zero_share_analysis"]), "dropped_missing_future": int(audit["dropped_missing_future"])}


def validate_matched_samples(samples: dict[str, set[str]]) -> pd.DataFrame:
    checks = []
    for lag, baseline in [("fsa_t_to_pa_tplus1", "matched_pa_tplus1_baseline"), ("fsa_t_to_pa_tplus2", "matched_pa_tplus2_baseline")]:
        left, right = samples[lag], samples[baseline]
        checks.append({"lag_specification": lag, "matched_baseline": baseline, "n_lag": len(left), "n_baseline": len(right), "intersection_n": len(left & right), "identical_samples": left == right, "lag_only_n": len(left - right), "baseline_only_n": len(right - left)})
    result = pd.DataFrame(checks)
    if not result["identical_samples"].all(): raise RuntimeError("Matched baseline sample check failed")
    return result


def build_report(output_dir: Path, effects: pd.DataFrame, sample_audit: pd.DataFrame, matches: pd.DataFrame) -> None:
    lines = ["# Treatment-margin and temporal DML analyses", "", f"Script version: `{SCRIPT_VERSION}`.", "", "## Design checks", "", "- All estimates use 3-fold random fully cross-fitted DML with seed 42, LassoCV nuisances, and HC3 inference.", "- Each FT-Sinkhorn result retrains one encoder per outer fold; held-out fold treatment and outcome are excluded from encoder training and tuning.", "- Binary extensive-margin FT-Sinkhorn uses a logistic (BCE-with-logits) treatment head and exact 0/1 Sinkhorn groups. Continuous models use four treatment quantile groups estimated from the training partition.", "- Temporal joins are by `stkcd` and exact calendar year. Encoder input is the anchor-date control vector X_t; future X is not supplied or imputed.", "- The intensive-margin estimand is conditional on raw FSA > 0 and differs from the full-sample continuous estimand.", "", "## Sample checks", ""]
    for _, row in matches.iterrows(): lines.append(f"- {row['lag_specification']} vs {row['matched_baseline']}: identical samples = {bool(row['identical_samples'])} (n = {int(row['n_lag'])}).")
    lines += ["", "## Estimates", "", effects.round(6).to_markdown(index=False), "", "## Sample audit", "", sample_audit.round(6).to_markdown(index=False), ""]
    (output_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--min-delta", type=float, default=0.0); parser.add_argument("--inner-valid-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=1024); parser.add_argument("--encode-batch-size", type=int, default=4096); parser.add_argument("--d-model", type=int, default=64); parser.add_argument("--n-heads", type=int, default=4); parser.add_argument("--n-layers", type=int, default=4); parser.add_argument("--mlp-hidden-dim", type=int, default=128); parser.add_argument("--latent-dim", type=int, default=32); parser.add_argument("--dropout", type=float, default=0.10); parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--treatment-groups", type=int, default=4); parser.add_argument("--outcome-weight", type=float, default=1.0); parser.add_argument("--treatment-weight", type=float, default=0.30); parser.add_argument("--sinkhorn-weight", type=float, default=0.10); parser.add_argument("--mmd-weight", type=float, default=0.10); parser.add_argument("--stability-weight", type=float, default=0.02); parser.add_argument("--perturb-std", type=float, default=0.02); parser.add_argument("--sinkhorn-epsilon", type=float, default=0.25); parser.add_argument("--sinkhorn-iters", type=int, default=20); parser.add_argument("--mmd-sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0]); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args(); core.validate_args(args)
    if args.folds != 3 or args.seed != 42: raise ValueError("This specification uses fixed settings: folds=3 and seed=42")
    if not args.output_name.strip() or Path(args.output_name).name != args.output_name: raise ValueError("--output-name must be one directory name")
    return args


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    output_dir = ROOT / "outputs" / args.output_name
    if output_dir.exists(): raise FileExistsError(f"Output directory exists: {output_dir}")
    output_dir.mkdir(parents=True); (output_dir / "runs").mkdir()
    snapshot = output_dir / "source_snapshot"; snapshot.mkdir()
    for source in [Path(__file__), Path(base.__file__), Path(core.__file__), HERE.parent / "src" / "config.py", HERE.parent / "src" / "data_utils.py"]:
        if source.exists(): shutil.copy2(source, snapshot / source.name)
    cfg = Config(); cfg.seed = args.seed; cfg.folds = args.folds
    raw = load_raw_panel(cfg)
    protocol = {"script_version": SCRIPT_VERSION, "seed": args.seed, "folds": args.folds, "args": vars(args), "raw_fsa": RAW_FSA, "outcome": PA, "controls": cfg.controls, "preprocessing": "calendar-year preprocessor fit through 2021; controls winsorized 1%/99%; treatment/outcome winsorized 1%/99% except binary treatment", "inference": "HC3", "nuisance": "LassoCV(cv=3)", "heldout_DY_used_for_encoder": False, "future_X_used": False, "future_imputation": False, "data_hashes": {"search": sha256(cfg.search_path), "innovation": sha256(cfg.innovation_path)}}
    write_json(output_dir / "run_protocol.json", protocol)
    margin_specs = ["baseline_continuous", "extensive_margin", "intensive_margin"]
    temporal_specs = ["fsa_t_to_pa_t", "fsa_t_to_pa_tplus1", "fsa_t_to_pa_tplus2", "fsa_tplus1_to_pa_t", "matched_pa_tplus1_baseline", "matched_pa_tplus2_baseline"]
    rows, audits, samples = [], [], {}
    for family, specs in [("margin", margin_specs), ("temporal", temporal_specs)]:
        for spec in specs:
            print(f"\n===== {family.upper()} {spec} =====", flush=True)
            frame, audit = (margin_design(raw, cfg, spec) if family == "margin" else temporal_design(raw, cfg, spec))
            bundle, ordered = prepare_dataset(frame, cfg), None
            ordered = ordered_from_bundle(bundle)
            if len(ordered) != len(frame): raise RuntimeError("Preprocessing row conservation failed")
            samples[spec] = set(ordered["anchor_key"])
            audit.update({"n_preprocess_train": int(len(bundle["train_df"])), "n_preprocess_valid": int(len(bundle["valid_df"])), "n_preprocess_test": int(len(bundle["test_df"])), "n_features": int(bundle["x_all"].shape[1]), "row_conservation_pass": True, "finite_arrays_pass": bool(np.isfinite(bundle["x_all"]).all() and np.isfinite(bundle["y_all"]).all() and np.isfinite(bundle["t_all"]).all())})
            spec_dir = output_dir / "runs" / spec
            raw_result = run_raw(bundle, ordered, args, spec_dir / "raw_x")
            brol_result = run_ft_sinkhorn_dml(bundle, ordered, args, spec_dir / "ft_sinkhorn", bool(audit["binary_treatment"]))
            Y, T = bundle["y_all"], bundle["t_all"]
            rows += [effect_row(family, spec, "Raw-X DML", raw_result, Y, T, audit), effect_row(family, spec, "BROL-FT-Sinkhorn DML", brol_result, Y, T, audit)]
            audits.append(audit)
            pd.DataFrame(rows).to_csv(output_dir / "all_effects.csv", index=False)
            pd.DataFrame(audits).to_csv(output_dir / "sample_audit.csv", index=False)
    effects, sample_audit = pd.DataFrame(rows), pd.DataFrame(audits)
    matches = validate_matched_samples(samples); matches.to_csv(output_dir / "matched_sample_checks.csv", index=False)
    effects.loc[effects["analysis_family"] == "margin"].to_csv(output_dir / "margin_results.csv", index=False)
    effects.loc[effects["analysis_family"] == "temporal"].to_csv(output_dir / "temporal_results.csv", index=False)
    fold_audits = []
    for path in (output_dir / "runs").glob("*/*/fold_audit.csv"):
        table = pd.read_csv(path); table.insert(0, "run_path", str(path.relative_to(output_dir))); fold_audits.append(table)
    pd.concat(fold_audits, ignore_index=True).to_csv(output_dir / "full_fold_audit.csv", index=False)
    build_report(output_dir, effects, sample_audit, matches)
    write_json(output_dir / "run_status.json", {"status": "completed", "n_specifications": len(margin_specs) + len(temporal_specs), "n_estimates": len(rows), "matched_samples_pass": True})
    print(f"Completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
