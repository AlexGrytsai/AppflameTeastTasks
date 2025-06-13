import settings  # noqa
from train_ml_model import DataPreparation, NLTKResources

if __name__ == "__main__":
    nltk_data = NLTKResources()
    data_preparation = DataPreparation()

    toxicity_list_column = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    nltk_data.download_nltk_resources()

    data_preparation.add_binary_label(toxicity_list_column)

    data_preparation.separate_data_into_train_and_test()
