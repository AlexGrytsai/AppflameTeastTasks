import logging
from typing import Union

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from train_ml_model import DataPreparation

logger = logging.getLogger(__name__)


class ModelML:
    def __init__(
        self,
        model: Union[LogisticRegression, MultinomialNB],
        data_for_training: DataPreparation,
    ) -> None:
        self.model = model.fit(
            data_for_training.x_train_vectorized, data_for_training.y_train
        )
        self.predict = self.model.predict(data_for_training.x_test_vectorized)
        self.data_for_training = data_for_training

    def get_assessment_model_performance(self):
        logger.info("Evaluating model performance")

        logger.info(
            "Classification report: \n%s",
            classification_report(self.data_for_training.y_test, self.predict),
        )
