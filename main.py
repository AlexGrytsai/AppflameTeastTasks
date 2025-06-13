from sklearn.linear_model import LogisticRegression

import settings  # noqa
from train_ml_model import DataPreparation, NLTKResources, ModelML

if __name__ == "__main__":
    nltk_data = NLTKResources()
    data_for_training = DataPreparation()

    toxicity_list_column = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    nltk_data.download_nltk_resources()

    data_for_training.add_binary_label(toxicity_list_column)

    data_for_training.separate_data_into_train_and_test()

    data_for_training.vectorize_data()

    # Logistic Regression model
    model_lr = ModelML(
        LogisticRegression(C=1, max_iter=1000),
        data_for_training,
    )
    model_lr.get_assessment_model_performance()
