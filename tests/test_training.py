from attrition_model.training import build_models


def test_models_produce_probabilities() -> None:
    features = __import__("pandas").DataFrame(
        {
            "satisfaction_level": [0.2, 0.8, 0.3, 0.9],
            "last_evaluation": [0.8, 0.7, 0.9, 0.6],
            "number_project": [6, 3, 5, 2],
            "average_monthly_hours": [250, 160, 230, 140],
            "time_spend_company": [5, 2, 4, 2],
            "Work_accident": [0, 1, 0, 1],
            "promotion_last_5years": [0, 1, 0, 1],
            "Department": ["sales", "technical", "sales", "technical"],
            "salary": ["low", "high", "medium", "high"],
        }
    )
    target = [1, 0, 1, 0]
    for model in build_models(forest_estimators=10).values():
        model.fit(features, target)
        probabilities = model.predict_proba(features)[:, 1]
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
