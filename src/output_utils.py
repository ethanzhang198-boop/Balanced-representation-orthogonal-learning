from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame], analysis_lines: list[str]):
    analysis_df = pd.DataFrame({"analysis": analysis_lines})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        analysis_df.to_excel(writer, sheet_name="analysis", index=False)
