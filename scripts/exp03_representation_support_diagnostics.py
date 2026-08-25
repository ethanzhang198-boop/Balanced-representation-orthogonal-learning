"""Entry point for representation-support diagnostics."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        choices=("k100", "common-x"),
        required=True,
        help="Select the k100 or common-x diagnostic.",
    )
    args = parser.parse_args()
    support = Path(__file__).resolve().parent / "_support"
    analysis_script = {
        "k100": support / "exp03c_k100_neighborhood_diagnostics.py",
        "common-x": support / "exp03a_common_x_matched_balance.py",
    }[args.analysis]
    sys.argv = [str(analysis_script)]
    runpy.run_path(str(analysis_script), run_name="__main__")


if __name__ == "__main__":
    main()
