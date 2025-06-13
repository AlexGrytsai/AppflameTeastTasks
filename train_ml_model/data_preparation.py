import logging
from typing import Optional, List

import nltk
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class DataPreparation:
    @staticmethod
    def download_nltk_resources(resources: Optional[List[str]] = None):
        """
        Downloads specified NLTK resources or a default set if none are
        provided.

        Parameters:
        resources (Optional[List[str]]): A list of NLTK resource names to
                                         download. If None, downloads
                                         a default set of resources including
                                         "punkt", "stopwords", and "wordnet".

        This method logs the download process and ensures that the required
        NLTK resources are available for text processing tasks.
        """
        if not resources:
            logger.info(
                "Resources not specified. Downloading base resources..."
            )
            resources = ["punkt", "stopwords", "wordnet"]
        for resource in resources:
            logger.info(f"Downloading resource: {resource}")
            nltk.download(resource)

    @staticmethod
    def download_data_for_training(path: str = "train.csv") -> DataFrame:
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
