import logging
import re
from typing import List, Set, Optional, Tuple

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(
        self,
        path_to_data_for_training: str = "train.csv",
        stop_words: Optional[Set[str]] = None,
        lemmatizer: Optional[WordNetLemmatizer] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ) -> None:
        self.training_data = self._download_data_for_training(
            path_to_data_for_training
        )
        self.stop_words = stop_words or set(stopwords.words("english"))
        self.lemmatizer = lemmatizer or WordNetLemmatizer()
        self.vectorizer = vectorizer or TfidfVectorizer()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_vectorized = None
        self.x_test_vectorized = None

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

    def vectorize_data(self) -> None:
        logger.info("Vectorizing data")

        x_train_preprocessed, x_test_preprocessed = self._preprocess_datasets()
        self.x_train_vectorized = self.vectorizer.fit_transform(
            x_train_preprocessed
        )
        self.x_test_vectorized = self.vectorizer.transform(x_test_preprocessed)

        logger.info("Data vectorized successfully")

    def _preprocess_datasets(self) -> Tuple[DataFrame, DataFrame]:
        logger.info("Preprocessing datasets")
        x_train_preprocessed = self.x_train.apply(self._preprocess_text)
        x_test_preprocessed = self.x_test.apply(self._preprocess_text)
        logger.info("Datasets preprocessed successfully")

        return x_train_preprocessed, x_test_preprocessed

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesses a string by cleaning it, tokenizing it, lemmatizing the
        tokens, and joining them back into a string.

        Parameters:
        text (str): The string to preprocess.

        Returns:
        str: The preprocessed string.
        """
        logger.info("Preprocessing text")
        clean_text = self._clean_text(text)
        tokens = self._tokenize_text(clean_text)
        lemmatized_tokens = self._lemmatize_text(tokens)
        logger.info("Text preprocessed successfully")

        return " ".join(lemmatized_tokens)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans a string by converting it to lowercase, removing URLs, special
        characters, and numbers.

        Parameters:
        text (str): The string to clean.

        Returns:
        str: The cleaned string.
        """
        text = text.lower()

        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\d+", "", text)

        return text

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """
        Tokenizes a string into a list of words using NLTK's word tokenizer.

        Parameters:
        text (str): The string to be tokenized.

        Returns:
        List[str]: A list of word tokens extracted from the input string.
        """
        return nltk.word_tokenize(text)

    def _lemmatize_text(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizes a list of word tokens using NLTK's WordNet lemmatizer.

        Parameters:
        tokens (List[str]): A list of word tokens to be lemmatized.

        Returns:
        List[str]: A list of lemmatized word tokens.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def _apply_text_preprocessing(self):
        self.x_train = self.x_train.apply(self._preprocess_text)
        self.x_test = self.x_test.apply(self._preprocess_text)
