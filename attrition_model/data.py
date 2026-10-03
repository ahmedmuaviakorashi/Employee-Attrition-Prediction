from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = {
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "left",
    "promotion_last_5years",
    "Department",
    "salary",
}


def load_employee_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = SOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("The employee dataset is empty.")
    if frame["left"].isna().any() or not set(frame["left"].unique()).issubset({0, 1}):
        raise ValueError("The target column 'left' must contain only 0 and 1.")
    if frame.isna().any().any():
        raise ValueError("The supplied capstone dataset is expected to contain no missing values.")
    return frame.rename(columns={"average_montly_hours": "average_monthly_hours"})


def prepare_modeling_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    repeated_after_first = int(frame.duplicated().sum())
    modeling_frame = frame.drop_duplicates().reset_index(drop=True)
    summary = {
        "source_rows": len(frame),
        "exact_repeated_rows_after_first": repeated_after_first,
        "modeling_rows": len(modeling_frame),
        "employees_who_stayed": int((modeling_frame["left"] == 0).sum()),
        "employees_who_left": int((modeling_frame["left"] == 1).sum()),
    }
    return modeling_frame, summary
