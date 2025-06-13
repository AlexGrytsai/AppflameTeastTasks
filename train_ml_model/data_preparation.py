import logging
from typing import Optional, List

import nltk

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
