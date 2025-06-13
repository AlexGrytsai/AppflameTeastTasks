import settings  # noqa
from train_ml_model import DataPreparation

if __name__ == "__main__":
    data_preparation = DataPreparation()

    data_preparation.download_nltk_resources()
