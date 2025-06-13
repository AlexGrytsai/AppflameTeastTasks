import settings  # noqa
from train_ml_model import DataPreparation, NLTKResources

if __name__ == "__main__":
    nltk_data = NLTKResources()
    data_preparation = DataPreparation()

    nltk_data.download_nltk_resources()
