from pathlib import Path

import pandas as pd
import pytest

from attrition_model.data import load_employee_data, prepare_modeling_data

DATASET = Path("Datasets/HR_capstone_dataset.csv")


def test_supplied_dataset_profile() -> None:
    frame = load_employee_data(DATASET)
    modeling_frame, summary = prepare_modeling_data(frame)
    assert summary == {
        "source_rows": 14999,
        "exact_repeated_rows_after_first": 3008,
        "modeling_rows": 11991,
        "employees_who_stayed": 10000,
        "employees_who_left": 1991,
    }
    assert "average_monthly_hours" in modeling_frame.columns


def test_rejects_invalid_target(tmp_path: Path) -> None:
    frame = pd.read_csv(DATASET).head(2)
    frame.loc[0, "left"] = 2
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only 0 and 1"):
        load_employee_data(path)
