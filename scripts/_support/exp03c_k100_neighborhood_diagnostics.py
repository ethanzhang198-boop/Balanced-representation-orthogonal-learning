from __future__ import annotations

"""k=100 neighborhood diagnostics for representation-support analyses."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import Config
from src.data_utils import prepare_dataset


SOURCE = ROOT / "outputs" / "exp06_within_year_permutation_placebo_100rep"
RECONSTRUCTION_SOURCE = (
    ROOT
    / "outputs"
    / "exp03_representation_support_diagnostics"
    / "common_x_matched_balance"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "exp03_representation_support_diagnostics"
    / "k100_neighborhood_diagnostics"
)
NN_K = int(os.environ.get("NN_K", "100"))


def fit_cutoffs(values: np.ndarray, n_groups: int) -> np.ndarray:
    return np.quantile(
        np.asarray(values, dtype=float),
        np.arange(1, n_groups, dtype=float) / n_groups,
    )


def apply_groups(values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return np.digitize(
        np.asarray(values, dtype=float),
        np.asarray(cutoffs, dtype=float),
        right=True,
    ).astype(int)


def pairwise_smd(x: np.ndarray, groups: np.ndarray) -> float:
    values: list[float] = []
    for left_group in range(4):
        for right_group in range(left_group + 1, 4):
            left = x[groups == left_group]
            right = x[groups == right_group]
            pooled_sd = np.sqrt((left.var(0, ddof=1) + right.var(0, ddof=1)) / 2)
            pooled_sd[pooled_sd < 1e-12] = 1.0
            values.append(np.abs((left.mean(0) - right.mean(0)) / pooled_sd).mean())
    return float(np.mean(values))


def summary_scores(
    space: str,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    train_geometry: np.ndarray,
    test_geometry: np.ndarray,
    k: int,
    treatment: np.ndarray,
) -> pd.DataFrame:
    _, neighbor_index = cKDTree(train_geometry).query(
        test_geometry,
        k=min(k, len(train_index)),
        workers=1,
    )
    neighbor_index = np.asarray(neighbor_index)
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, None]

    cross_quartile = (train_groups[neighbor_index] != test_groups[:, None]).mean(1)
    decile_cutoffs = fit_cutoffs(treatment[train_index], 10)
    train_decile = apply_groups(treatment[train_index], decile_cutoffs)
    test_decile = apply_groups(treatment[test_index], decile_cutoffs)
    same_decile = (train_decile[neighbor_index] == test_decile[:, None]).mean(1)
    middle_decile = ((train_decile[neighbor_index] >= 1) & (train_decile[neighbor_index] <= 8)).mean(1)

    return pd.DataFrame(
        {
            "space": space,
            "outer_fold": fold,
            "row_index": test_index,
            "treatment_q4": test_groups,
            "treatment_decile": test_decile,
            "q4_cross_share": cross_quartile,
            "middle80_neighbor_share": middle_decile,
            "decile_same_share": same_decile,
        }
    )


def build_tables(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table4_rows: list[dict[str, object]] = []
    table5_rows: list[dict[str, object]] = []
    table6_rows: list[dict[str, object]] = []
    for space, group in scores.groupby("space", sort=False):
        cross = group.q4_cross_share
        table4_rows.append(
            {
                "space": space,
                "mean_cross_quartile_neighbor_share": cross.mean(),
                "cross_quartile_share_ge_0_25": (cross >= 0.25).mean(),
                "cross_quartile_share_ge_0_50": (cross >= 0.50).mean(),
                "low_overlap_share_cross_lt_0_25": (cross < 0.25).mean(),
                "severe_low_overlap_share_cross_lt_0_10": (cross < 0.10).mean(),
                "nobs": len(group),
            }
        )
        for quartile, label in [(0, "Bottom quartile"), (3, "Top quartile")]:
            subset = group[group.treatment_q4 == quartile]
            cross = subset.q4_cross_share
            table5_rows.append(
                {
                    "space": space,
                    "tail": label,
                    "mean_nonown_quartile_neighbor_share": cross.mean(),
                    "common_support_share_cross_ge_0_20": (cross >= 0.20).mean(),
                    "isolated_share_cross_lt_0_10": (cross < 0.10).mean(),
                    "nobs": len(subset),
                }
            )
        for decile, label in [(0, "Bottom decile"), (9, "Top decile")]:
            subset = group[group.treatment_decile == decile]
            middle = subset.middle80_neighbor_share
            table6_rows.append(
                {
                    "space": space,
                    "tail": label,
                    "middle80_neighbor_share": middle.mean(),
                    "middle80_support_share_ge_0_50": (middle >= 0.50).mean(),
                    "isolated_tail_share_same_tail_gt_0_75": (subset.decile_same_share > 0.75).mean(),
                    "nobs": len(subset),
                }
            )
    return pd.DataFrame(table4_rows), pd.DataFrame(table5_rows), pd.DataFrame(table6_rows)


def main() -> None:
    source_arrays = SOURCE / "runs" / "base_ft_sinkhorn" / "oof_arrays.npz"
    if not source_arrays.exists():
        raise FileNotFoundError(
            "Missing full-crossfit OOF arrays. Run "
            "scripts/exp06_within_year_permutation_placebo.py first."
        )
    coordinate_paths = [
        RECONSTRUCTION_SOURCE / f"recreated_fold_{fold}_query_reference_z.npz"
        for fold in [1, 2, 3]
    ]
    missing_coordinates = [path for path in coordinate_paths if not path.exists()]
    if missing_coordinates:
        raise FileNotFoundError(
            "Missing common-X fold coordinates. Run "
            "scripts/exp03_representation_support_diagnostics.py "
            "--analysis common-x before --analysis k100."
        )

    bundle = prepare_dataset(Config())
    x = np.asarray(bundle["x_all"], dtype=float)
    treatment = np.asarray(bundle["t_all"], dtype=float)
    print("Loaded panel", flush=True)

    fold_id = np.load(source_arrays)["fold_id"]
    fold_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    for fold in [1, 2, 3]:
        print(f"Fold {fold}", flush=True)
        train_index = np.flatnonzero(fold_id != fold)
        test_index = np.flatnonzero(fold_id == fold)
        quartile_cutoffs = fit_cutoffs(treatment[train_index], 4)
        train_groups = apply_groups(treatment[train_index], quartile_cutoffs)
        test_groups = apply_groups(treatment[test_index], quartile_cutoffs)

        raw_scaler = StandardScaler()
        raw_train = raw_scaler.fit_transform(x[train_index])
        raw_test = raw_scaler.transform(x[test_index])
        raw_pca = PCA(n_components=10, whiten=True, random_state=42 + fold)
        raw_train = raw_pca.fit_transform(raw_train)
        raw_test = raw_pca.transform(raw_test)

        coordinates = np.load(coordinate_paths[fold - 1])
        representation_train = coordinates["z_reference"]
        representation_test = coordinates["z_query"]
        representation_scaler = StandardScaler()
        representation_train = representation_scaler.fit_transform(representation_train)
        representation_test = representation_scaler.transform(representation_test)
        representation_pca = PCA(n_components=10, whiten=True, random_state=42 + fold)
        representation_train = representation_pca.fit_transform(representation_train)
        representation_test = representation_pca.transform(representation_test)

        for space, train_geometry, test_geometry in [
            ("Raw-X", raw_train, raw_test),
            ("BROL-FT-Sinkhorn (reconstructed)", representation_train, representation_test),
        ]:
            print(f"Fold {fold}: {space} neighbors", flush=True)
            scores = summary_scores(
                space,
                fold,
                train_index,
                test_index,
                train_groups,
                test_groups,
                train_geometry,
                test_geometry,
                NN_K,
                treatment,
            )
            score_frames.append(scores)
            fold_rows.append(
                {
                    "space": space,
                    "outer_fold": fold,
                    "n_test": len(test_index),
                    "smd_pair_mean": pairwise_smd(test_geometry, test_groups),
                    "cross_group_neighbor_share_mean": scores.q4_cross_share.mean(),
                    "low_overlap_share_cross_lt_0_25": (scores.q4_cross_share < 0.25).mean(),
                    "severe_low_overlap_share_cross_lt_0_10": (scores.q4_cross_share < 0.10).mean(),
                }
            )

    scores = pd.concat(score_frames, ignore_index=True)
    fold_summary = pd.DataFrame(fold_rows)
    table4, table5, table6 = build_tables(scores)
    stored_table3 = pd.read_csv(SOURCE / "table3.csv")
    table3_rows: list[dict[str, object]] = []
    for space in fold_summary.space.unique():
        subset = fold_summary[fold_summary.space == space]
        weights = subset.n_test
        if space == "Raw-X":
            record = stored_table3[stored_table3.space == "Raw-X"].iloc[0].to_dict()
        else:
            record = stored_table3[stored_table3.space.str.startswith("BROL-FT")].iloc[0].to_dict()
            record["space"] = space
        record.update(
            {
                "nobs": int(weights.sum()),
                "smd_pair_mean": float(np.average(subset.smd_pair_mean, weights=weights)),
                "cross_group_neighbor_share_mean": float(np.average(subset.cross_group_neighbor_share_mean, weights=weights)),
                "low_overlap_share_cross_lt_0_25": float(np.average(subset.low_overlap_share_cross_lt_0_25, weights=weights)),
                "severe_low_overlap_share_cross_lt_0_10": float(np.average(subset.severe_low_overlap_share_cross_lt_0_10, weights=weights)),
                "nn_k": NN_K,
            }
        )
        table3_rows.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table3_rows).to_csv(OUTPUT_DIR / f"table3_k{NN_K}.csv", index=False)
    table4.to_csv(OUTPUT_DIR / f"table4_k{NN_K}.csv", index=False)
    table5.to_csv(OUTPUT_DIR / f"table5_k{NN_K}.csv", index=False)
    table6.to_csv(OUTPUT_DIR / f"table6_k{NN_K}.csv", index=False)
    fold_summary.to_csv(OUTPUT_DIR / f"table3_k{NN_K}_fold_detail.csv", index=False)
    scores.to_csv(OUTPUT_DIR / f"k{NN_K}_geometry_scores.csv", index=False)
    pd.concat(
        [
            pd.read_csv(SOURCE / f"table{i}.csv").assign(version="stored_k50")
            for i in [3, 4, 5, 6]
        ],
        keys=["Table3", "Table4", "Table5", "Table6"],
    ).to_csv(OUTPUT_DIR / "stored_k50_reference.csv")
    (OUTPUT_DIR / "README.md").write_text(
        "k=50/k=100 neighborhood diagnostics. The protocol uses held-out queries "
        "and same-fold outer-training references. Raw-X reproduces the stored k=50 "
        "neighbor/support values. BROL uses CPU-reconstructed fold encoders because "
        "checkpoints and fold-training Z arrays are unavailable; compare the k=100 "
        "and k=50 BROL values only within this reconstruction. MMD, Sinkhorn, and "
        "OOF R2 do not depend on k; SMD and neighbor/support quantities are "
        "recomputed from the available geometry.\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
