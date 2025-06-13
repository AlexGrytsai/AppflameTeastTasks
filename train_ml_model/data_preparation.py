import logging
from typing import List

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(
        self,
        path_to_data_for_training: str = "train.csv",
    ) -> None:
        self.training_data = self._download_data_for_training(
            path_to_data_for_training
        )

    @staticmethod
    def _download_data_for_training(path: str) -> DataFrame:
        """
        Downloads a CSV file containing data for training the model, and
        returns a Pandas DataFrame with the downloaded data.

        Parameters:
        path (str): The path to the CSV file to download. Defaults to
                    "train.csv".

        Returns:
        DataFrame: A Pandas DataFrame containing the downloaded data.
        """
        return pd.read_csv(path)

    def add_binary_label(self, toxicity_list_column: List[str]) -> None:
        """
        Creates a new binary label column in the training data from a list of
        toxicity-related columns.

        The binary label is created by summing the values in the specified
        columns and checking if the sum is greater than 0. If the sum is
        greater than 0, the binary label is set to 1, otherwise it is set to 0.

        Parameters:
        toxicity_list_column (List[str]): A list of column names to use for
                                          creating the binary label.

        Returns:
        None
        """
        logger.info("Adding binary label")
        self.training_data["toxic_binary"] = (
            self.training_data[toxicity_list_column].sum(axis=1) > 0
        ).astype(int)
        logger.info(f"Total number of toxic labels: {len(self.training_data)}")
