from pathlib import Path

import pandas as pd


def create_submission(
    test_ids,
    predictions,
    output_path,
    id_column="id",
    label_column="label"
):
    """
    Function to create and save a prediction submission CSV.

    Parameters:
        test_ids: IDs from the test dataset, in their original order.
        predictions: Predicted labels corresponding to the test IDs.
        output_path: Location where the CSV should be saved.
        id_column: Name of the ID column.
        label_column: Name of the prediction column.

    """

    # Reset the indices without changing the row order
    test_ids = pd.Series(test_ids).reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)

    # Every test ID must have exactly one prediction
    if len(test_ids) != len(predictions):
        raise ValueError(
            "Number of test IDs does not match number of predictions."
        )

    if test_ids.isnull().any():
        raise ValueError("The test IDs contain missing values.")

    if predictions.isnull().any():
        raise ValueError("The predictions contain missing values.")

    submission_df = pd.DataFrame({
        id_column: test_ids,
        label_column: predictions
    })

    output_path = Path(output_path)

    # Create the submissions folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # index=False prevents pandas from adding an unwanted column
    submission_df.to_csv(output_path, index=False)

    return submission_df