import logging
from typing import Optional, List

import nltk
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(
        self,
        path_to_data_for_training: str = "train.csv",
        nltk_resources: Optional[List[str]] = None,
    ) -> None:
        self.path_to_data_for_training = path_to_data_for_training
        self.nltk_resources = (
            nltk_resources
            if nltk_resources is not None
            else ["punkt", "stopwords", "wordnet"]
        )

    @staticmethod
    def download_nltk_resources(resources: List[str]):
        """
        Downloads specified NLTK resources.

        Parameters:
        resources (Optional[List[str]]): A list of NLTK resource names to
                                         download.

        This method logs the download process and ensures that the required
        NLTK resources are available for text processing tasks.
        """
        for resource in resources:
            logger.info(f"Downloading resource: {resource}")
            nltk.download(resource)

    @staticmethod
    def download_data_for_training(path: str) -> DataFrame:
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
