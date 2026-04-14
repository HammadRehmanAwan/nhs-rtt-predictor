"""Dataset loading and validation utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    Columns,
    DATASET_PATHS,
    FULL_HISTORY_TARGET_MONTHS,
    KAGGLE_DATASET_SLUG,
    KAGGLE_JSON_FILENAME,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)


class DataLoadError(RuntimeError):
    """Raised when the application cannot load RTT data."""


class DataValidationError(ValueError):
    """Raised when the dataset is missing required columns or values."""


@dataclass(frozen=True)
class DataLoadResult:
    """Loaded RTT data and basic metadata."""

    dataframe: pd.DataFrame
    source_path: str
    source_label: str
    period_count: int


def _count_distinct_periods(path: Path) -> int:
    """Count distinct `period` values without loading the full dataset."""

    try:
        period_frame = pd.read_csv(path, usecols=[Columns.PERIOD], low_memory=False)
    except Exception:
        return 0
    return int(period_frame[Columns.PERIOD].dropna().nunique())


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the dataframe contains the minimum columns used by the app."""

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise DataValidationError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )


def _candidate_paths(dataset_paths: Iterable[str] | None = None) -> list[Path]:
    paths = DATASET_PATHS if dataset_paths is None else dataset_paths
    return [Path(path) for path in paths]


def find_local_dataset(dataset_paths: Iterable[str] | None = None) -> Path | None:
    """Return the most complete readable local dataset path."""

    ranked_paths: list[tuple[int, Path]] = []
    for path in _candidate_paths(dataset_paths):
        if path.exists() and path.is_file():
            ranked_paths.append((_count_distinct_periods(path), path))
    if not ranked_paths:
        return None
    ranked_paths.sort(key=lambda item: (item[0], item[1].stat().st_size), reverse=True)
    return ranked_paths[0][1]


def _normalise_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    validate_required_columns(df)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df[Columns.PERIOD_DT] = pd.to_datetime(
        df[Columns.PERIOD],
        format="%Y-%m",
        errors="coerce",
    )
    df = df.dropna(subset=[Columns.PERIOD_DT]).sort_values(Columns.PERIOD_DT)

    if df.empty:
        raise DataValidationError(
            "Dataset did not contain any valid monthly period values after parsing."
        )

    return df


def load_local_dataset(path: Path) -> DataLoadResult:
    """Load RTT data from a local CSV file."""

    try:
        dataframe = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Local dataset not found at {path}") from exc
    except Exception as exc:
        raise DataLoadError(f"Could not read local dataset at {path}: {exc}") from exc

    return DataLoadResult(
        dataframe=_normalise_dataframe(dataframe),
        source_path=str(path),
        source_label="Local CSV",
        period_count=_count_distinct_periods(path),
    )


def _write_kaggle_credentials(credentials: dict[str, str]) -> Path:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    credentials_path = kaggle_dir / KAGGLE_JSON_FILENAME
    with credentials_path.open("w", encoding="utf-8") as file_handle:
        json.dump(credentials, file_handle)
    os.chmod(credentials_path, 0o600)

    return credentials_path


def download_dataset_from_kaggle(
    credentials: dict[str, str],
    download_dir: str = "/tmp",
    dataset_slug: str = KAGGLE_DATASET_SLUG,
) -> Path:
    """Download the CSV from Kaggle and return the discovered file path."""

    if not credentials.get("username") or not credentials.get("key"):
        raise DataLoadError("Kaggle credentials must include username and key.")

    _write_kaggle_credentials(credentials)
    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "--unzip",
        "-p",
        str(target_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "Unknown Kaggle error."
        raise DataLoadError(f"Kaggle download failed: {stderr}")

    csv_files = sorted(target_dir.glob("*.csv"))
    if not csv_files:
        raise DataLoadError(
            f"Kaggle download completed, but no CSV file was found in {target_dir}."
        )

    return csv_files[0]


def load_rtt_data(
    dataset_paths: Iterable[str] | None = None,
    allow_kaggle_download: bool = False,
    kaggle_credentials: dict[str, str] | None = None,
    download_dir: str = "/tmp",
) -> DataLoadResult:
    """Load RTT data, preferring a local CSV and only using Kaggle as fallback."""

    local_path = find_local_dataset(dataset_paths)
    if local_path is not None:
        return load_local_dataset(local_path)

    if allow_kaggle_download and kaggle_credentials:
        downloaded_path = download_dataset_from_kaggle(
            credentials=kaggle_credentials,
            download_dir=download_dir,
        )
        result = load_local_dataset(downloaded_path)
        return DataLoadResult(
            dataframe=result.dataframe,
            source_path=result.source_path,
            source_label="Kaggle download",
            period_count=result.period_count,
        )

    searched_paths = ", ".join(str(path) for path in _candidate_paths(dataset_paths))
    raise DataLoadError(
        "No RTT dataset was found locally. "
        f"Searched: {searched_paths}. "
        "Place the CSV in the project folder or configure optional Kaggle credentials."
    )


def prefer_fuller_dataset(
    local_result: DataLoadResult | None,
    kaggle_result: DataLoadResult | None,
    minimum_full_history_months: int = FULL_HISTORY_TARGET_MONTHS,
) -> DataLoadResult | None:
    """Choose the richer dataset when both local and Kaggle sources are available."""

    if local_result is None:
        return kaggle_result
    if kaggle_result is None:
        return local_result

    if local_result.period_count >= minimum_full_history_months:
        return local_result
    if kaggle_result.period_count > local_result.period_count:
        return kaggle_result
    return local_result
