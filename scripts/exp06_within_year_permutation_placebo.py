from __future__ import annotations

"""Full-crossfit diagnostics and within-year permutation analysis."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _support.full_crossfit_pipeline as base
from src.data_utils import build_preprocessor, split_panel


HERE = Path(__file__).resolve().parent
SCRIPT_VERSION = "FULL_CROSSFIT_WITHIN_YEAR_PERMUTATION_20260802"
DEFAULT_OUTPUT_NAME = "exp06_within_year_permutation_placebo_100rep"
EMPTY_VALID_POLICY = (
    "Allow a zero-row preprocessing validation bucket only when the "
    "leave-one-year-out specification removes Config.valid_year; never call "
    "sklearn transformers on that empty frame."
)


def prepare_year_dataset(
    config: Any,
    df: pd.DataFrame,
    *,
    allow_expected_empty_valid: bool,
) -> dict[str, Any]:

    frame = df.copy()
    feature_cols = [c for c in config.controls if c in frame.columns] + ["year"]
    numeric_cols = [c for c in config.controls if c in frame.columns]
    categorical_cols = ["year"]
    train_df, valid_df, test_df = split_panel(
        frame,
        config.train_end,
        config.valid_year,
    )

    if train_df.empty:
        raise RuntimeError("Table 12 preprocessing training bucket is empty")
    if test_df.empty:
        raise RuntimeError("Table 12 preprocessing test bucket is empty")
    if valid_df.empty and not allow_expected_empty_valid:
        raise RuntimeError(
            "Table 12 preprocessing validation bucket is unexpectedly empty"
        )
    if allow_expected_empty_valid and not valid_df.empty:
        raise RuntimeError(
            "The expected-empty validation rule was enabled, but validation is nonempty"
        )

    preprocessor = build_preprocessor(feature_cols, numeric_cols, categorical_cols)
    preprocessor.fit(train_df[feature_cols])

    def transform(nonempty_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = preprocessor.transform(nonempty_frame[feature_cols])
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        y = nonempty_frame[config.outcome].to_numpy(dtype=np.float32)
        t = nonempty_frame[config.treatment].to_numpy(dtype=np.float32)
        return x, y, t

    x_train, y_train, t_train = transform(train_df)
    if valid_df.empty:
        x_valid = np.empty((0, x_train.shape[1]), dtype=np.float32)
        y_valid = np.empty(0, dtype=np.float32)
        t_valid = np.empty(0, dtype=np.float32)
    else:
        x_valid, y_valid, t_valid = transform(valid_df)
    x_test, y_test, t_test = transform(test_df)

    x_all = np.vstack([x_train, x_valid, x_test])
    y_all = np.concatenate([y_train, y_valid, y_test])
    t_all = np.concatenate([t_train, t_valid, t_test])
    if len(x_all) != len(frame):
        raise RuntimeError(
            f"Table 12 row conservation failed: transformed={len(x_all)}, frame={len(frame)}"
        )
    if not (
        np.isfinite(x_all).all()
        and np.isfinite(y_all).all()
        and np.isfinite(t_all).all()
    ):
        raise RuntimeError("Table 12 transformed arrays contain non-finite values")

    return {
        "df": frame,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "x_train": x_train,
        "y_train": y_train,
        "t_train": t_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "t_valid": t_valid,
        "x_test": x_test,
        "y_test": y_test,
        "t_test": t_test,
        "x_all": x_all,
        "y_all": y_all,
        "t_all": t_all,
    }


def run_fingerprint_with_empty_validation(
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
        "wrapper_source_hash": base.file_sha256(Path(__file__).resolve()),
        "base_runner_source_hash": base.file_sha256(Path(base.__file__).resolve()),
        "core_source_hash": base.file_sha256(Path(base.core.__file__).resolve()),
        "empty_valid_policy": EMPTY_VALID_POLICY,
        "run_id": run_id,
        "spec": spec,
        "split_kind": split_kind,
        "seed": args.seed,
        "folds": args.folds,
        "model_args": base.model_arg_payload(args),
        "geometry": geometry,
        "heterogeneity": heterogeneity,
        "include_raw_geometry": include_raw_geometry,
        "data_hash": base.array_sha256(
            np.asarray(bundle["x_all"]),
            np.asarray(bundle["y_all"]),
            np.asarray(bundle["t_all"]),
        ),
    }
    return base.payload_sha256(payload), payload


def stage_year_with_empty_validation(output_dir: Path, args: argparse.Namespace) -> None:
    cfg = base.Config()
    cfg.seed = args.seed
    cfg.folds = args.folds
    raw_merged = base.load_raw_merged_panel(cfg)
    years = sorted(int(value) for value in raw_merged["year"].dropna().unique())
    year_specs: list[tuple[str, set[int]]] = [
        (f"Exclude {year}", {year}) for year in years
    ]
    year_specs.append(("Exclude 2020-2021", {2020, 2021}))
    if args.smoke:
        year_specs = [(f"Exclude {cfg.valid_year}", {int(cfg.valid_year)})]

    rows: list[dict[str, Any]] = []
    split_audit: list[dict[str, Any]] = []
    for label, excluded in year_specs:
        df = base.panel_from_raw_merged(cfg, raw_merged, excluded)
        allow_expected_empty_valid = int(cfg.valid_year) in excluded
        bundle = prepare_year_dataset(
            cfg,
            df,
            allow_expected_empty_valid=allow_expected_empty_valid,
        )
        audit_row = {
            "specification": label,
            "excluded_years": ",".join(map(str, sorted(excluded))),
            "n_remaining": int(len(df)),
            "n_preprocess_train": int(len(bundle["train_df"])),
            "n_preprocess_valid": int(len(bundle["valid_df"])),
            "n_preprocess_test": int(len(bundle["test_df"])),
            "n_features": int(bundle["x_all"].shape[1]),
            "expected_empty_valid_allowed": bool(allow_expected_empty_valid),
            "row_conservation_pass": int(len(bundle["x_all"])) == int(len(df)),
            "finite_arrays_pass": bool(
                np.isfinite(bundle["x_all"]).all()
                and np.isfinite(bundle["y_all"]).all()
                and np.isfinite(bundle["t_all"]).all()
            ),
        }
        split_audit.append(audit_row)
        pd.DataFrame(split_audit).to_csv(
            output_dir / "table12_split_audit.csv",
            index=False,
        )
        print(
            "[YEAR SPLIT] "
            f"excluded={audit_row['excluded_years']} "
            f"train={audit_row['n_preprocess_train']} "
            f"valid={audit_row['n_preprocess_valid']} "
            f"test={audit_row['n_preprocess_test']} "
            f"allow_empty_valid={allow_expected_empty_valid}",
            flush=True,
        )

        ordered_df = base.ordered_from_bundle(bundle)
        token = "_".join(map(str, sorted(excluded)))
        result = base.run_representation_crossfit(
            f"year_exclude_{token}",
            bundle,
            ordered_df,
            base.FT_SINKHORN_SPEC,
            "random",
            output_dir,
            args,
        )
        rows.append(
            {
                "specification": label,
                "excluded_years": audit_row["excluded_years"],
                "year_removed_before_preprocessing": True,
                "representation_retrained": True,
                "n_preprocess_train": audit_row["n_preprocess_train"],
                "n_preprocess_valid": audit_row["n_preprocess_valid"],
                "n_preprocess_test": audit_row["n_preprocess_test"],
                "expected_empty_valid_allowed": allow_expected_empty_valid,
                **result["effect"],
            }
        )
        pd.DataFrame(rows).to_csv(output_dir / "table12.csv", index=False)


def write_protocol_with_empty_validation(
    output_dir: Path,
    args: argparse.Namespace,
    stages: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(base.core.__file__).resolve(),
        (base.ROOT / "src" / "data_utils.py").resolve(),
        (base.ROOT / "src" / "config.py").resolve(),
    ]
    for source in source_paths:
        shutil.copy2(source, snapshot / source.name)

    import econml
    import sklearn

    base.write_json(
        output_dir / "run_protocol.json",
        {
            "script_version": SCRIPT_VERSION,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": base.file_sha256(Path(__file__).resolve()),
            "base_runner_source": str(Path(base.__file__).resolve()),
            "base_runner_sha256": base.file_sha256(Path(base.__file__).resolve()),
            "core_model_source": str(Path(base.core.__file__).resolve()),
            "core_model_sha256": base.file_sha256(Path(base.core.__file__).resolve()),
            "empty_valid_policy": EMPTY_VALID_POLICY,
            "empty_valid_scope": "Table 12 only; expected only for Exclude 2022",
            "smoke_boundary_case": "Exclude 2022",
            "python_executable": str(Path(__import__("sys").executable).resolve()),
            "scikit_learn_version": sklearn.__version__,
            "econml_version": econml.__version__,
            "model_rule": "BROL-FT-Sinkhorn from the shared full-crossfit script",
            "common_z_used": False,
            "fixed_z_used_for_inference": False,
            "heldout_DY_used_for_encoder": False,
            "stages": stages,
            "args": vars(args),
            "existing_output_directories_modified": False,
        },
    )


def install_entrypoint_overrides() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.DEFAULT_OUTPUT_NAME = DEFAULT_OUTPUT_NAME
    base.run_fingerprint = run_fingerprint_with_empty_validation
    base.stage_year = stage_year_with_empty_validation
    base.write_protocol = write_protocol_with_empty_validation


def main() -> None:
    if "--placebo-reps" not in sys.argv:
        sys.argv.extend(["--placebo-reps", "100"])
    install_entrypoint_overrides()
    base.main()


if __name__ == "__main__":
    main()
