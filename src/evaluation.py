import numpy as np


def calculate_macro_f1(y_true, y_pred):
    """
    Fuction calculates the macro F1 score for the given true and predicted labels.

    Parameters:
        y_true: The actual labels.
        y_pred: The labels predicted by the model.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = []
    for label in (0, 1):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(scores))
