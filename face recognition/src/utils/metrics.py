from statistics import mean


def classification_metrics(y_true: list[str], y_pred: list[str], confidences: list[float] | None = None) -> dict:
    if not y_true:
        return {
            "accuracy": 0.0,
            "unknown_rate": 0.0,
            "known_prediction_rate": 0.0,
            "known_only_accuracy": 0.0,
            "mean_confidence": None,
        }

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    unknown_predictions = sum(int(pred == "unknown") for pred in y_pred)
    known_predictions = len(y_pred) - unknown_predictions
    known_correct = sum(
        int(t == p and p != "unknown")
        for t, p in zip(y_true, y_pred)
    )

    return {
        "accuracy": correct / len(y_true),
        "unknown_rate": unknown_predictions / len(y_pred),
        "known_prediction_rate": known_predictions / len(y_pred),
        "known_only_accuracy": (known_correct / known_predictions) if known_predictions else 0.0,
        "mean_confidence": mean(confidences) if confidences else None,
    }


def paired_privacy_metrics(pred_orig: list[str], pred_anon: list[str]) -> dict:
    if not pred_orig:
        return {
            "prediction_agreement_rate": 0.0,
            "prediction_change_rate": 0.0,
            "anonymized_unknown_rate": 0.0,
        }

    agreements = sum(int(a == b) for a, b in zip(pred_orig, pred_anon))
    anon_unknown = sum(int(pred == "unknown") for pred in pred_anon)
    total = len(pred_orig)
    return {
        "prediction_agreement_rate": agreements / total,
        "prediction_change_rate": 1.0 - (agreements / total),
        "anonymized_unknown_rate": anon_unknown / total,
    }
