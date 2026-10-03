import argparse
import json
from pathlib import Path

from attrition_model.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate attrition models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Datasets/HR_capstone_dataset.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--forest-estimators", type=int, default=400)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    result = run_training(
        args.data,
        args.output,
        forest_estimators=args.forest_estimators,
        cv_folds=args.cv_folds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
