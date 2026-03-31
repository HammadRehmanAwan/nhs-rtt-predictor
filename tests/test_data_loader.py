from __future__ import annotations

import pandas as pd
import pytest

from src.config import Columns
from src.data_loader import (
    DataValidationError,
    load_local_dataset,
    validate_required_columns,
)


def test_validate_required_columns_raises_for_missing_columns() -> None:
    dataframe = pd.DataFrame({Columns.PERIOD: ["2024-01"]})

    with pytest.raises(DataValidationError):
        validate_required_columns(dataframe)


def test_load_local_dataset_normalises_periods(tmp_path, sample_rtt_df) -> None:
    csv_path = tmp_path / "sample.csv"
    sample_rtt_df.drop(columns=[Columns.PERIOD_DT]).to_csv(csv_path, index=False)

    result = load_local_dataset(csv_path)

    assert result.source_label == "Local CSV"
    assert result.source_path == str(csv_path)
    assert Columns.PERIOD_DT in result.dataframe.columns
    assert result.dataframe[Columns.PERIOD_DT].isna().sum() == 0
    assert list(result.dataframe[Columns.PERIOD].unique()) == ["2024-01", "2024-02"]
