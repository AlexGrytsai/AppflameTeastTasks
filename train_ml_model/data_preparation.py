import logging

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
        self._add_binary_label()

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

    def _add_binary_label(self) -> None:
        """
        Creates a binary label by summing the number of toxic labels
        (toxic, severe_toxic, obscene, threat, insult, identity_hate) and
        checking if the sum is greater than 0. The result is stored in
        the 'toxic_binary' column of the training data.

        Returns:
            None
        """
        self.training_data["toxic_binary"] = (
            self.training_data[
                [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                ]
            ].sum(axis=1)
            > 0
        ).astype(int)
