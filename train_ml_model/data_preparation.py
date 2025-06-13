import logging
from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(
        self,
        path_to_data_for_training: str = "train.csv",
    ) -> None:
        self.training_data = self._download_data_for_training(
            path_to_data_for_training
        )
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

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

    def separate_data_into_train_and_test(
        self,
        test_size: float = 0.2,
        random_state=42,
    ) -> None:
        """
        Separates the training data into training and test sets using
        scikit-learn's `train_test_split`.

        The `test_size` parameter determines the proportion of the data to
        include in the test set. The `random_state` parameter is used for
        reproducibility of the split.

        The function logs the shapes of the training and test data.

        Parameters:
        test_size (float): The proportion of the data to include in the test
                           set. Defaults to 0.2.
        random_state (int): The seed used to shuffle the data. Defaults to 42.

        Returns:
        None
        """
        logger.info("Separating data into train and test")

        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(
                self.training_data["comment_text"],
                self.training_data["toxic_binary"],
                test_size=test_size,
                random_state=random_state,
            )
        )

        logger.info(f"Train data shape: {self.x_train.shape[0]}")
        logger.info(f"Test data shape: {self.x_test.shape[0]}")
