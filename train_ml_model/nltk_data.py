import logging
from typing import List, Optional

import nltk

logger = logging.getLogger(__name__)


class NLTKResources:
    def __init__(self, nltk_resources: Optional[List[str]] = None) -> None:
        self.nltk_resources = (
            nltk_resources
            if nltk_resources is not None
            else ["punkt", "stopwords", "wordnet"]
        )

    @staticmethod
    def download_nltk_resources(resources: List[str]) -> bool:
        """
        Downloads specified NLTK resources.

        Parameters:
        resources (Optional[List[str]]): A list of NLTK resource names to
                                         download.

        This method logs the download process and ensures that the required
        NLTK resources are available for text processing tasks.
        """
        try:
            for resource in resources:
                logger.info(f"Downloading resource: {resource}")
                nltk.download(resource)
            logger.info("NLTK resources downloaded successfully.")
            return True
        except Exception as exc:
            logger.error(f"Failed to download NLTK resource. Exception: {exc}")
            raise exc
