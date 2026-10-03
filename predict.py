import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one employee record.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--record", required=True, help="Employee features as a JSON object.")
    args = parser.parse_args()
    model = joblib.load(args.model)
    frame = pd.DataFrame([json.loads(args.record)])
    probability = float(model.predict_proba(frame)[:, 1][0])
    print(json.dumps({"probability_of_leaving": probability}, indent=2))


if __name__ == "__main__":
    main()
