import math
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
from sklearn.decomposition import TruncatedSVD


class RecommenderSVD:
    """Collaborative filtering recommender based on SVD."""

    def __init__(self, random_seed: Optional[int] = None):
        """Create SVD recommender.

        :param random_seed: random seed.
        """
        self._random_seed = random_seed
        self._chunk_size = 1000000
        self._user_embeds = None
        self._item_embeds = None

    def fit(self, interactions: scipy.sparse.coo_matrix,
            embed_size: int = 100) -> None:
        """Build user and item embeddings.

        :param interactions: interaction matrix.
        :param embed_size: size of embeddings or dimension of the latent space.
        """
        # Create model
        svd = TruncatedSVD(n_components=embed_size,
                           random_state=self._random_seed)

        # Train
        transformed = svd.fit_transform(interactions)

        # Extract X, Sigma, and YT
        x = transformed / svd.singular_values_
        sigma = np.diag(svd.singular_values_)
        yt = svd.components_

        # Compose user and item embeddings
        sigma_sqrt = np.sqrt(sigma)
        self._user_embeds = np.dot(x, sigma_sqrt)
        self._item_embeds = np.dot(sigma_sqrt, yt)

    def predict(self, user_work_ids: Tuple[np.ndarray, np.ndarray]) \
            -> np.ndarray:
        """Calculate predicted rating.

        :param user_work_ids: user and work ids which ratings to predict.
        :return: predicted ratings.
        """
        if (self._user_embeds is None) or (self._item_embeds is None):
            raise ValueError('The model must be fitted before predicting.')
        user_ids, work_ids = user_work_ids
        samples_count = user_ids.shape[0]

        # If there are not so many samples
        if samples_count <= self._chunk_size:
            return np.sum(self._user_embeds[user_ids, :]
                          * self._item_embeds[:, work_ids].T, axis=1)

        # If there are many samples, we need to reduce memory consumption
        result = np.zeros(samples_count)
        chunks_count = math.ceil(samples_count / self._chunk_size)
        for chunk in np.array_split(range(samples_count), chunks_count):
            user_ids_chunk = user_ids[chunk]
            work_ids_chunk = work_ids[chunk]
            result[chunk] = np.sum(self._user_embeds[user_ids_chunk, :]
                                   * self._item_embeds[:, work_ids_chunk].T,
                                   axis=1)
        return result
