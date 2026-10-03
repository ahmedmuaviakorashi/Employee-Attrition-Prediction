import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition_model.data import load_employee_data, prepare_modeling_data

plt.switch_backend("Agg")

TARGET = "left"
CATEGORICAL_FEATURES = ["Department", "salary"]
NUMERIC_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_monthly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(random_state: int = 42, forest_estimators: int = 400) -> dict[str, Any]:
    return {
        "prevalence_baseline": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=forest_estimators,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _test_metrics(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
    }


def _cross_validation_metrics(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int,
    random_state: int,
) -> dict[str, float]:
    validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model,
        features,
        target,
        cv=validation,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
        n_jobs=-1,
    )
    return {
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
        "average_precision_mean": float(scores["test_average_precision"].mean()),
        "average_precision_std": float(scores["test_average_precision"].std()),
    }


def _plot_target(target: pd.Series, output: Path) -> None:
    counts = target.value_counts().sort_index()
    ax = counts.rename(index={0: "Stayed", 1: "Left"}).plot.bar(
        color=["#2a9d8f", "#e76f51"], figsize=(6, 4)
    )
    ax.set_title("Modeling population by outcome")
    ax.set_xlabel("")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=0)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(output, dpi=160)
    plt.close(ax.get_figure())


def _plot_curves(
    models: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series, output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in models.items():
        probabilities = model.predict_proba(features)[:, 1]
        false_positive, true_positive, _ = roc_curve(target, probabilities)
        precision, recall, _ = precision_recall_curve(target, probabilities)
        label = name.replace("_", " ").title()
        axes[0].plot(false_positive, true_positive, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_confusion_matrix(
    model: Pipeline, features: pd.DataFrame, target: pd.Series, output: Path
) -> None:
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        features,
        target,
        display_labels=["Stayed", "Left"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Random-forest test predictions at threshold 0.50")
    display.figure_.tight_layout()
    display.figure_.savefig(output, dpi=160)
    plt.close(display.figure_)


def _feature_reliance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        features,
        target,
        scoring="average_precision",
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def _plot_feature_reliance(frame: pd.DataFrame, output: Path) -> None:
    ordered = frame.sort_values("importance_mean")
    ax = ordered.plot.barh(
        x="feature",
        y="importance_mean",
        xerr="importance_std",
        legend=False,
        color="#457b9d",
        figsize=(8, 5),
    )
    ax.set_title("Random-forest permutation importance")
    ax.set_xlabel("Decrease in average precision")
    ax.set_ylabel("")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(output, dpi=160)
    plt.close(ax.get_figure())


def _report(data_summary: dict[str, int], metrics: dict[str, Any]) -> str:
    logistic = metrics["logistic_regression"]["test"]
    forest = metrics["random_forest"]["test"]
    return f"""# Employee attrition model results

## Data used

- Supplied rows: **{data_summary['source_rows']:,}**
- Exact repeated rows after the first occurrence: **{data_summary['exact_repeated_rows_after_first']:,}**
- Distinct rows used for modeling: **{data_summary['modeling_rows']:,}**
- Stayed: **{data_summary['employees_who_stayed']:,}**
- Left: **{data_summary['employees_who_left']:,}**

The dataset contains no employee identifier. Exact repeated rows could be duplicate records or different employees with identical de-identified values. The primary analysis removes repeated rows to prevent identical records from appearing in both training and test sets; this is a modeling safeguard, not a claim that the people are duplicates.

## Held-out test performance

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | {logistic['roc_auc']:.3f} | {logistic['average_precision']:.3f} | {logistic['precision']:.3f} | {logistic['recall']:.3f} | {logistic['f1']:.3f} | {logistic['accuracy']:.3f} |
| Random forest | {forest['roc_auc']:.3f} | {forest['average_precision']:.3f} | {forest['precision']:.3f} | {forest['recall']:.3f} | {forest['f1']:.3f} | {forest['accuracy']:.3f} |

ROC-AUC and average precision use predicted probabilities, not hard class labels. Cross-validation summaries are available in `metrics.json` and were calculated on the training partition only.

## Interpretation and responsible use

Permutation importance describes which inputs the fitted random forest relies on; it does not establish why employees leave. Satisfaction, evaluation, workload, and tenure may reflect current organizational conditions, may be unavailable at the intended prediction time, and may be influenced by earlier management decisions.

This model should support aggregate investigation, voluntary retention programs, and workload review. It should not be used as the sole basis for decisions about an individual employee, compensation, promotion, discipline, or termination. Deployment would require a defined prediction time, protected-attribute and subgroup audits, monitoring, employee-data governance, and human review.
"""


def run_training(
    data_path: Path,
    output_directory: Path,
    random_state: int = 42,
    forest_estimators: int = 400,
    cv_folds: int = 5,
) -> dict[str, Any]:
    source = load_employee_data(data_path)
    frame, data_summary = prepare_modeling_data(source)
    features = frame.drop(columns=TARGET)
    target = frame[TARGET]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )

    models = build_models(random_state, forest_estimators)
    metrics: dict[str, Any] = {}
    for name, model in models.items():
        metrics[name] = {
            "cross_validation": _cross_validation_metrics(
                model, train_features, train_target, cv_folds, random_state
            )
        }
        model.fit(train_features, train_target)
        metrics[name]["test"] = _test_metrics(model, test_features, test_target)

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    reliance = _feature_reliance(
        models["random_forest"], test_features, test_target, random_state
    )
    reliance.to_csv(output_directory / "feature_reliance.csv", index=False)
    payload = {
        "data": data_summary,
        "split": {
            "train_rows": len(train_features),
            "test_rows": len(test_features),
            "test_fraction": 0.2,
            "random_state": random_state,
        },
        "models": metrics,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_directory / "model_report.md").write_text(
        _report(data_summary, metrics), encoding="utf-8"
    )
    _plot_target(target, figure_directory / "target_distribution.png")
    _plot_curves(models, test_features, test_target, figure_directory / "model_curves.png")
    _plot_confusion_matrix(
        models["random_forest"],
        test_features,
        test_target,
        figure_directory / "confusion_matrix.png",
    )
    _plot_feature_reliance(reliance, figure_directory / "feature_reliance.png")
    joblib.dump(models["random_forest"], output_directory / "model.joblib")
    return payload
