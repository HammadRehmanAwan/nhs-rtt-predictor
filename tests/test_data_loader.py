from __future__ import annotations

import pandas as pd
import pytest

from src.config import Columns
from src.data_loader import (
    DataLoadResult,
    DataValidationError,
    choose_fullest_dataset,
    extract_google_drive_file_id,
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


def test_extract_google_drive_file_id_from_share_url() -> None:
    file_id = extract_google_drive_file_id(
        "https://drive.google.com/file/d/1qhxhC72HM208UZmzEcR7S_vWf1ywRsAQ/view?usp=drive_link"
    )

    assert file_id == "1qhxhC72HM208UZmzEcR7S_vWf1ywRsAQ"


def test_choose_fullest_dataset_prefers_more_periods(sample_rtt_df) -> None:
    small = DataLoadResult(sample_rtt_df, "small.csv", "Local CSV", period_count=2)
    fuller = DataLoadResult(sample_rtt_df, "full.csv", "Google Drive", period_count=60)

    chosen = choose_fullest_dataset([small, fuller])

    assert chosen == fuller
