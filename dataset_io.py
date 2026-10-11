"""Чтение таблиц ground_truth.csv, в том числе сохранённых русским Excel."""
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = ("length_mm", "width_mm", "marker_mm")


def read_table(path: Path) -> pd.DataFrame:
    """Читает CSV и сам определяет разделитель (',' или ';'); '48,5' превращает в 48.5.

    Excel с русской локалью сохраняет CSV с разделителем ';' и десятичной запятой —
    обычное чтение при этом даёт одну «склеенную» колонку, и данные теряются без ошибки.
    """
    table = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python", dtype=str)
    table.columns = [str(c).strip() for c in table.columns]
    for col in NUMERIC_COLUMNS:
        if col in table.columns:
            table[col] = pd.to_numeric(
                table[col].astype(str).str.strip().str.replace(",", ".", regex=False)
                .replace({"": None, "nan": None}), errors="coerce")
    for col in table.columns:
        if col not in NUMERIC_COLUMNS and table[col].dtype == object:
            table[col] = table[col].astype(str).str.strip().replace({"nan": ""})
    return table
