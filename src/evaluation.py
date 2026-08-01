from sklearn.metrics import f1_score


def calculate_macro_f1(y_true, y_pred):
    """
    Fuction calculates the macro F1 score for the given true and predicted labels.

    Parameters:
        y_true: The actual labels.
        y_pred: The labels predicted by the model.
    """
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )