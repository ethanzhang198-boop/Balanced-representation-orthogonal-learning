from __future__ import annotations

"""Fold-specific common-X matched-balance diagnostics."""

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
from src.config import Config
from src.data_utils import prepare_dataset


DEFAULT_SOURCE = ROOT / "outputs" / "exp06_within_year_permutation_placebo_100rep"
DEFAULT_OUTPUT = ROOT / "outputs" / "exp03_representation_support_diagnostics" / "common_x_matched_balance"


@dataclass(frozen=True)
class MatchingSpec:
    name: str
    k: int
    allocation: str


SPECS = [
    MatchingSpec("pooled_nonown_k25", 25, "pooled_nonown"),
    MatchingSpec("pooled_nonown_k50", 50, "pooled_nonown"),
    MatchingSpec("pooled_nonown_k100", 100, "pooled_nonown"),
    MatchingSpec("balanced_other_quartiles_k25", 25, "balanced_other_quartiles"),
    MatchingSpec("balanced_other_quartiles_k50", 50, "balanced_other_quartiles"),
    MatchingSpec("balanced_other_quartiles_k100", 100, "balanced_other_quartiles"),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mmd-sample-size", type=int, default=600)
    p.add_argument("--skip-encoder-fidelity-check", action="store_true")
    p.add_argument("--folds-to-run", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--encode-fold-only", action="store_true")
    return p


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def q_groups(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return np.asarray(core.apply_quantile_groups(values, cutoffs), dtype=np.int64)


def allocation_counts(k: int, n_groups: int = 3) -> list[int]:
    return [k // n_groups + (int(i < k % n_groups)) for i in range(n_groups)]


def choose_neighbors(
    geometry_query: np.ndarray,
    geometry_reference: np.ndarray,
    q_group: np.ndarray,
    r_group: np.ndarray,
    spec: MatchingSpec,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    def exact_top_k(query_values: np.ndarray, candidate_values: np.ndarray, candidate_index: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        out_index: list[np.ndarray] = []
        out_distance: list[np.ndarray] = []
        for start in range(0, len(query_values), 8):
            query_chunk = query_values[start : start + 8]
            delta = query_chunk[:, None, :] - candidate_values[None, :, :]
            squared = np.sum(delta * delta, axis=2)
            local = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
            selected = np.take_along_axis(squared, local, axis=1)
            order = np.argsort(selected, axis=1)
            out_index.append(candidate_index[np.take_along_axis(local, order, axis=1)])
            out_distance.append(np.sqrt(np.take_along_axis(selected, order, axis=1)))
        return np.vstack(out_index), np.vstack(out_distance)

    n_query = len(q_group)
    neighbours: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_query)]
    distances: list[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(n_query)]
    all_groups = sorted(np.unique(r_group).tolist())
    for group in sorted(np.unique(q_group).tolist()):
        q_idx = np.flatnonzero(q_group == group)
        other = [g for g in all_groups if g != group]
        if spec.allocation == "pooled_nonown":
            cand = np.flatnonzero(r_group != group)
            use_k = min(spec.k, len(cand))
            idx, dist = exact_top_k(geometry_query[q_idx], geometry_reference[cand], cand, use_k)
            for local, global_idx in enumerate(q_idx):
                neighbours[global_idx] = idx[local]
                distances[global_idx] = dist[local]
        elif spec.allocation == "balanced_other_quartiles":
            parts = allocation_counts(spec.k, len(other))
            per_group_idx: list[np.ndarray] = []
            per_group_dist: list[np.ndarray] = []
            for target_group, target_k in zip(other, parts):
                cand = np.flatnonzero(r_group == target_group)
                use_k = min(target_k, len(cand))
                idx, dist = exact_top_k(geometry_query[q_idx], geometry_reference[cand], cand, use_k)
                per_group_idx.append(idx)
                per_group_dist.append(dist)
            for local, global_idx in enumerate(q_idx):
                neighbours[global_idx] = np.concatenate([part[local] for part in per_group_idx])
                distances[global_idx] = np.concatenate([part[local] for part in per_group_dist])
        else:
            raise ValueError(spec.allocation)
    return neighbours, distances


def median_bandwidth(x_train_standardized: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(x_train_standardized)
    pairs = min(20_000, n * (n - 1) // 2)
    left = rng.integers(0, n, size=pairs)
    right = rng.integers(0, n, size=pairs)
    unequal = left != right
    while not np.all(unequal):
        right[~unequal] = rng.integers(0, n, size=(~unequal).sum())
        unequal = left != right
    diff = x_train_standardized[left] - x_train_standardized[right]
    median = float(np.median(np.sqrt(np.einsum("ij,ij->i", diff, diff))))
    return max(median, 1e-8)


def weighted_rbf_mmd(
    query: np.ndarray,
    reference: np.ndarray,
    ref_weights: np.ndarray,
    bandwidth: float,
    seed: int,
    max_n: int,
) -> float:
    if len(query) < 2 or len(reference) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    nq = min(len(query), max_n)
    q_idx = rng.choice(len(query), nq, replace=False)
    weights = np.asarray(ref_weights, dtype=np.float64)
    weights = weights / weights.sum()
    nr = min(len(reference), max_n)
    r_idx = rng.choice(len(reference), nr, replace=True, p=weights)
    q = query[q_idx]
    r = reference[r_idx]
    def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = left[:, None, :] - right[None, :, :]
        return np.einsum("ijk,ijk->ij", delta, delta)
    dqq = squared_distances(q, q)
    drr = squared_distances(r, r)
    dqr = squared_distances(q, r)
    denom = 2.0 * bandwidth * bandwidth
    kqq = np.exp(-dqq / denom)
    krr = np.exp(-drr / denom)
    kqr = np.exp(-dqr / denom)
    return float(kqq.mean() + krr.mean() - 2.0 * kqr.mean())


def weighted_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(np.median(values)), float(np.quantile(values, 0.90))


def panel_mask(panel: str, q4: np.ndarray) -> np.ndarray:
    if panel == "Full held-out sample":
        return np.ones(len(q4), dtype=bool)
    if panel == "Bottom treatment quartile / zero-heavy lower tail":
        return q4 == 0
    if panel == "Top treatment quartile":
        return q4 == 3
    raise ValueError(panel)


PANELS = [
    "Full held-out sample",
    "Bottom treatment quartile / zero-heavy lower tail",
    "Top treatment quartile",
]


def summarize_rule(
    *,
    fold: int,
    panel: str,
    rule: str,
    spec: MatchingSpec,
    xq: np.ndarray,
    xr: np.ndarray,
    qg: np.ndarray,
    rg: np.ndarray,
    selections: list[np.ndarray],
    precision: np.ndarray,
    bandwidth: float,
    mmd_sample_size: int,
    seed: int,
    n_reference: int,
) -> dict[str, Any]:
    mask = panel_mask(panel, qg)
    active = np.flatnonzero(mask)
    if not len(active):
        raise RuntimeError(f"No query observations in {panel}, fold {fold}.")
    smd_cells: list[np.ndarray] = []
    mmd_values: list[tuple[int, float]] = []
    euclidean: list[np.ndarray] = []
    mahalanobis: list[np.ndarray] = []
    neighbour_counts: list[int] = []
    for group_a in sorted(np.unique(qg[active]).tolist()):
        q_idx = active[qg[active] == group_a]
        for group_b in sorted(np.unique(rg).tolist()):
            if group_b == group_a:
                continue
            ref_parts: list[np.ndarray] = []
            for idx in q_idx:
                selected = selections[int(idx)]
                ref_parts.append(selected[rg[selected] == group_b])
            picked = np.concatenate(ref_parts) if ref_parts else np.empty(0, dtype=np.int64)
            if not len(picked):
                continue
            query_values = xq[q_idx]
            ref_values = xr[picked]
            weights = np.bincount(picked, minlength=len(xr)).astype(np.float64)
            weights = weights / weights.sum()
            matched_mean = np.sum(xr * weights[:, None], axis=0)
            smd_cells.append(np.abs(query_values.mean(axis=0) - matched_mean))
            mmd = weighted_rbf_mmd(
                query_values,
                xr,
                weights,
                bandwidth,
                seed + 1000 * fold + 100 * group_a + group_b + spec.k,
                mmd_sample_size,
            )
            mmd_values.append((len(q_idx), mmd))
        for idx in q_idx:
            selected = selections[int(idx)]
            if not len(selected):
                continue
            delta = xr[selected] - xq[int(idx)]
            euclidean.append(np.sqrt(np.einsum("ij,ij->i", delta, delta)))
            diagonal_precision = np.diag(precision)
            mahalanobis.append(np.sqrt(np.sum(delta * delta * diagonal_precision[None, :], axis=1)))
            neighbour_counts.append(len(selected))
    if not smd_cells or not euclidean:
        raise RuntimeError(f"No valid matched samples for {panel}, {rule}, fold {fold}.")
    smd = np.concatenate(smd_cells)
    euc = np.concatenate(euclidean)
    mah = np.concatenate(mahalanobis)
    mmd_num = sum(n * value for n, value in mmd_values if np.isfinite(value))
    mmd_den = sum(n for n, value in mmd_values if np.isfinite(value))
    mean_euc, median_euc, p90_euc = weighted_quantiles(euc)
    mean_mah, median_mah, p90_mah = weighted_quantiles(mah)
    return {
        "outer_fold": fold,
        "panel": panel,
        "selection_rule": rule,
        "matching_spec": spec.name,
        "neighbor_allocation": spec.allocation,
        "k_requested": spec.k,
        "n_query": int(len(active)),
        "n_reference": int(n_reference),
        "mean_actual_neighbors": float(np.mean(neighbour_counts)),
        "min_actual_neighbors": int(np.min(neighbour_counts)),
        "mean_abs_smd_x": float(np.mean(smd)),
        "median_abs_smd_x": float(np.median(smd)),
        "max_abs_smd_x": float(np.max(smd)),
        "share_abs_smd_x_lt_0_10": float(np.mean(smd < 0.10)),
        "share_abs_smd_x_lt_0_20": float(np.mean(smd < 0.20)),
        "mean_x_euclidean_distance": mean_euc,
        "median_x_euclidean_distance": median_euc,
        "p90_x_euclidean_distance": p90_euc,
        "mean_x_mahalanobis_distance": mean_mah,
        "median_x_mahalanobis_distance": median_mah,
        "p90_x_mahalanobis_distance": p90_mah,
        "x_space_weighted_rbf_mmd": float(mmd_num / mmd_den) if mmd_den else float("nan"),
        "mmd_bandwidth_outer_training_median_rule": float(bandwidth),
        "n_smd_cells": int(len(smd)),
        "n_pairwise_distances": int(len(euc)),
    }


def pooled_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    continuous = [
        "mean_actual_neighbors", "mean_abs_smd_x", "median_abs_smd_x", "max_abs_smd_x",
        "mean_x_euclidean_distance", "median_x_euclidean_distance", "p90_x_euclidean_distance",
        "mean_x_mahalanobis_distance", "median_x_mahalanobis_distance", "p90_x_mahalanobis_distance",
        "x_space_weighted_rbf_mmd", "mmd_bandwidth_outer_training_median_rule",
    ]
    proportions = ["share_abs_smd_x_lt_0_10", "share_abs_smd_x_lt_0_20"]
    rows: list[dict[str, Any]] = []
    keys = ["panel", "selection_rule", "matching_spec", "neighbor_allocation", "k_requested"]
    for values, group in fold_results.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        w = group["n_query"].to_numpy(dtype=float)
        row["n_query"] = int(w.sum())
        row["n_reference_min"] = int(group["n_reference"].min())
        row["n_reference_max"] = int(group["n_reference"].max())
        for col in continuous:
            valid = np.isfinite(group[col].to_numpy(dtype=float))
            row[col] = float(np.average(group.loc[valid, col], weights=w[valid])) if valid.any() else float("nan")
        for col in proportions:
            num = (group[col].to_numpy(dtype=float) * group["n_smd_cells"].to_numpy(dtype=float)).sum()
            den = group["n_smd_cells"].sum()
            row[col] = float(num / den) if den else float("nan")
        row["n_smd_cells"] = int(group["n_smd_cells"].sum())
        row["n_pairwise_distances"] = int(group["n_pairwise_distances"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def folded_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table3 = pd.read_csv(source / "table3.csv")
    table4 = pd.read_csv(source / "table4.csv")
    table5 = pd.read_csv(source / "table5.csv")
    table6 = pd.read_csv(source / "table6.csv")
    geom = pd.read_csv(source / "runs" / "base_ft_sinkhorn" / "geometry_fold_summary.csv")
    raw_geom = geom[geom["space"] == "Raw-X"].copy()
    brol_geom = geom[geom["space"] == "BROL-FT-Sinkhorn (fold-specific)"].copy()
    table3_fold = pd.concat([brol_geom, raw_geom], ignore_index=True)
    scores = pd.read_csv(source / "runs" / "base_ft_sinkhorn" / "geometry_oof_scores.csv")
    score_rows: list[dict[str, Any]] = []
    for (space, fold), group in scores.groupby(["space", "outer_fold"], sort=False):
        cross = group["q4_cross_share"]
        score_rows.append({"table": "Table 4", "space": space, "outer_fold": fold, "tail": "All", "mean_cross_quartile_neighbor_share": cross.mean(), "low_overlap_share_cross_lt_0_25": (cross < .25).mean(), "severe_low_overlap_share_cross_lt_0_10": (cross < .10).mean(), "nobs": len(group)})
        for q, tail in [(0, "Bottom quartile"), (3, "Top quartile")]:
            sub = group[group["treatment_q4"] == q]
            score_rows.append({"table": "Table 5", "space": space, "outer_fold": fold, "tail": tail, "mean_nonown_quartile_neighbor_share": sub["q4_cross_share"].mean(), "common_support_share_cross_ge_0_20": (sub["q4_cross_share"] >= .20).mean(), "isolated_share_cross_lt_0_10": (sub["q4_cross_share"] < .10).mean(), "nobs": len(sub)})
        for d, tail in [(0, "Bottom decile"), (9, "Top decile")]:
            sub = group[group["treatment_decile"] == d]
            score_rows.append({"table": "Table 6", "space": space, "outer_fold": fold, "tail": tail, "middle80_neighbor_share": sub["middle80_neighbor_share"].mean(), "middle80_support_share_ge_0_50": (sub["middle80_neighbor_share"] >= .50).mean(), "isolated_tail_share_same_tail_gt_0_75": (sub["decile_same_share"] > .75).mean(), "nobs": len(sub)})
    table4_6_fold = pd.DataFrame(score_rows)
    pooled = pd.concat([
        table3.assign(table="Table 3"), table4.assign(table="Table 4"), table5.assign(table="Table 5"), table6.assign(table="Table 6"),
    ], ignore_index=True, sort=False)
    return table3, table4, table5, table6, table3_fold, table4_6_fold, pooled


def model_args(metadata: dict[str, Any], device: str) -> argparse.Namespace:
    payload = dict(metadata["fingerprint_payload"]["model_args"])
    payload.update({"seed": 42, "device": device})
    return argparse.Namespace(**payload)


def main(args: argparse.Namespace) -> None:
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    brol_path = source / "runs" / "base_ft_sinkhorn" / "oof_arrays.npz"
    raw_path = source / "base" / "raw_random_oof.npz"
    complete_path = source / "runs" / "base_ft_sinkhorn" / "complete.json"
    if not all(path.exists() for path in [brol_path, raw_path, complete_path]):
        raise FileNotFoundError("Saved BROL OOF arrays, Raw-X OOF arrays, and BROL metadata are required.")

    metadata = json.loads(complete_path.read_text(encoding="utf-8"))
    bundle = prepare_dataset(Config())
    x = np.asarray(bundle["x_all"], dtype=np.float64)
    treatment = np.asarray(bundle["t_all"], dtype=np.float64)
    brol_saved = np.load(brol_path)
    raw_saved = np.load(raw_path)
    fold_id = np.asarray(brol_saved["fold_id"], dtype=np.int64)
    if not np.array_equal(fold_id, np.asarray(raw_saved["fold_id"], dtype=np.int64)):
        raise RuntimeError("Raw-X and BROL OOF fold IDs differ.")
    splitter = KFold(n_splits=3, shuffle=True, random_state=args.seed)
    splits = [(tr, te) for tr, te in splitter.split(x)]
    for fold, (_, te) in enumerate(splits, start=1):
        if not np.array_equal(np.sort(te), np.flatnonzero(fold_id == fold)):
            raise RuntimeError("Saved OOF fold IDs do not reproduce the declared random outer split.")

    config = {
        "outer_folds": 3,
        "query_set": "held-out observations in each outer fold",
        "reference_set": "corresponding outer-training observations",
        "treatment_groups": 4,
        "cutoffs": "estimated only from outer-training D and applied to query and reference",
        "reference_spec": asdict(MatchingSpec("pooled_nonown_k50", 50, "pooled_nonown")),
        "sensitivity_specs": [asdict(item) for item in SPECS],
        "raw_x_geometry": "outer-training StandardScaler; Euclidean",
        "brol_z_geometry": "same fold-specific encoder for query/reference; row-L2-normalized; Euclidean",
        "evaluation_space": "outer-training-standardized original X",
        "mahalanobis": "LedoitWolf covariance fit only on outer-training standardized X",
        "mmd": {"kernel": "RBF", "bandwidth": "outer-training pooled median-distance rule", "weighted": True, "max_sample_per_side": args.mmd_sample_size},
    }
    (output / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    fit_args = model_args(metadata, args.device)
    spec = core.SPEC_LIBRARY["ft_sinkhorn"]
    fold_rows: list[dict[str, Any]] = []
    encoder_fidelity_rows: list[dict[str, Any]] = []
    fold_counts: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(splits, start=1):
        if fold not in args.folds_to_run:
            continue
        print(f"[COMMON-X] recreating fold-specific BROL encoder {fold}/3", flush=True)
        model, _, meta = core.fit_encoder_on_outer_training_fold(
            X_train=x[tr].astype(np.float32), Y_train=np.asarray(bundle["y_all"])[tr], T_train=treatment[tr],
            spec=spec, args=fit_args, seed=args.seed + 10_000 * fold, outer_fold=fold,
        )
        device = core.resolve_device(args.device)
        z_ref = core.encode_model(model, x[tr].astype(np.float32), fit_args.encode_batch_size, device)
        z_query = core.encode_model(model, x[te].astype(np.float32), fit_args.encode_batch_size, device)
        print(f"[COMMON-X] fold {fold}: encoded query/reference; checking saved OOF fidelity", flush=True)
        saved = np.asarray(brol_saved["z_oof"])[te]
        encoder_fidelity_rows.append({
            "outer_fold": fold, "n_query": len(te), "recreated_best_epoch": meta["best_epoch"],
            "saved_best_epoch": int(pd.read_csv(source / "runs" / "base_ft_sinkhorn" / "fold_audit.csv").query("outer_fold == @fold").iloc[0]["best_epoch"]),
            "max_abs_difference_recreated_vs_saved_oof_z": float(np.max(np.abs(z_query - saved))),
            "mean_abs_difference_recreated_vs_saved_oof_z": float(np.mean(np.abs(z_query - saved))),
            "cosine_similarity_mean_recreated_vs_saved_oof_z": float(np.mean(np.sum(l2_normalize(z_query) * l2_normalize(saved), axis=1))),
        })
        np.savez_compressed(
            output / f"recreated_fold_{fold}_query_reference_z.npz",
            train_index=tr,
            test_index=te,
            z_reference=z_ref,
            z_query=z_query,
        )
        pd.DataFrame([encoder_fidelity_rows[-1]]).to_csv(
            output / f"encoder_recreation_fidelity_fold_{fold}.csv", index=False
        )
        if args.encode_fold_only:
            print(f"[COMMON-X] wrote fold {fold} query/reference Z and will exit without matching", flush=True)
            continue
        del model
        gc.collect()
        scaler = StandardScaler().fit(x[tr])
        xr = scaler.transform(x[tr])
        xq = scaler.transform(x[te])
        precision = LedoitWolf().fit(xr).precision_
        cutoffs = core.fit_quantile_cutoffs(treatment[tr], 4)
        rg = q_groups(treatment[tr], cutoffs)
        qg = q_groups(treatment[te], cutoffs)
        bandwidth = median_bandwidth(xr, args.seed + 3000 * fold)
        fold_counts.append({
            "outer_fold": fold, "n_query": len(te), "n_reference": len(tr),
            "query_bottom_q4": int((qg == 0).sum()), "query_top_q4": int((qg == 3).sum()),
            "reference_bottom_q4": int((rg == 0).sum()), "reference_top_q4": int((rg == 3).sum()),
            "q1_cutoff": float(cutoffs[0]), "q2_cutoff": float(cutoffs[1]), "q3_cutoff": float(cutoffs[2]),
            "query_zero_share": float((treatment[te] == 0).mean()), "reference_zero_share": float((treatment[tr] == 0).mean()),
        })
        spaces = [("Raw-X-selected cross-treatment neighbors", xq, xr), ("BROL-Z-selected cross-treatment neighbors", l2_normalize(z_query), l2_normalize(z_ref))]
        for match_spec in SPECS:
            for rule, q_geometry, r_geometry in spaces:
                print(f"[COMMON-X] fold {fold}: {match_spec.name}; {rule}", flush=True)
                selections, _ = choose_neighbors(q_geometry, r_geometry, qg, rg, match_spec)
                for panel in PANELS:
                    fold_rows.append(summarize_rule(
                        fold=fold, panel=panel, rule=rule, spec=match_spec, xq=xq, xr=xr,
                        qg=qg, rg=rg, selections=selections, precision=precision, bandwidth=bandwidth,
                        mmd_sample_size=args.mmd_sample_size, seed=args.seed, n_reference=len(tr),
                    ))
        del z_ref, z_query
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fidelity = pd.DataFrame(encoder_fidelity_rows)
    if args.encode_fold_only:
        fidelity.to_csv(output / "encoder_recreation_fidelity.csv", index=False)
        print(f"Wrote encoded fold artefacts to {output}", flush=True)
        return
    if not args.skip_encoder_fidelity_check and set(args.folds_to_run) == {1, 2, 3}:
        if (fidelity["recreated_best_epoch"] != fidelity["saved_best_epoch"]).any() or (fidelity["max_abs_difference_recreated_vs_saved_oof_z"] > 1e-5).any():
            raise RuntimeError("Recreated fold encoders do not reproduce saved OOF Z; no common-X output was accepted.")
    fold_results = pd.DataFrame(fold_rows)
    pooled_results = pooled_summary(fold_results)
    reference_pooled = pooled_results[pooled_results["matching_spec"] == "pooled_nonown_k50"].copy()
    table3, table4, table5, table6, table3_fold, table4_6_fold, tables_pooled = folded_tables(source)
    tables_fold = pd.concat([table3_fold.assign(table="Table 3"), table4_6_fold], ignore_index=True, sort=False)
    comparisons: list[pd.DataFrame] = []
    for name, table in [("Table 3", table3), ("Table 4", table4), ("Table 5", table5), ("Table 6", table6)]:
        numeric = table.select_dtypes(include=[np.number]).columns.tolist()
        id_cols = [col for col in table.columns if col not in numeric]
        for _, row in table.iterrows():
            for col in numeric:
                comparisons.append(pd.DataFrame([{"table": name, **{key: row[key] for key in id_cols}, "metric": col, "stored_value": row[col], "current_value": row[col], "difference": 0.0, "reason": "Stored diagnostics use held-out queries and same-fold outer-training references."}]))
    comparison = pd.concat(comparisons, ignore_index=True)

    tables_pooled.to_csv(output / "tables3_6_pooled_diagnostics.csv", index=False)
    tables_fold.to_csv(output / "tables3_6_fold_level_diagnostics.csv", index=False)
    fold_results.to_csv(output / "common_x_matched_balance_fold_level.csv", index=False)
    pooled_results.to_csv(output / "common_x_matched_balance_pooled.csv", index=False)
    reference_pooled.to_csv(output / "common_x_matched_balance_reference.csv", index=False)
    fidelity.to_csv(output / "encoder_recreation_fidelity.csv", index=False)
    pd.DataFrame(fold_counts).to_csv(output / "fold_query_reference_counts.csv", index=False)
    comparison.to_csv(output / "tables3_6_stored_value_check.csv", index=False)
    for name, table in [("table3.csv", table3), ("table4.csv", table4), ("table5.csv", table5), ("table6.csv", table6)]:
        table.to_csv(output / name, index=False)
    print(f"Completed audit and common-X matching outputs in {output}", flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
