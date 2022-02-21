from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

from models.mixins import RatingScaleMixin


class RecommenderSVD(RatingScaleMixin):
    """Collaborative filtering recommender using matrix factorization approach
    based on Singular Value Decomposition.

    :param embed_size: size of embeddings or dimension of the latent space.
    :param rating_normalization: which rating normalization strategy
    to use ("mean", "z-score", or None)
    """

    def __init__(self, embed_size: int = 100,
                 rating_normalization: Optional[str] = 'mean') -> None:
        # Save arguments
        super().__init__(rating_normalization=rating_normalization)
        self.embed_size = embed_size
        self.rating_normalization = rating_normalization

        # To convert IDs of users and items into their indices
        self._item_categories = None
        self._user_categories = None

        # Embeddings
        self._user_embeddings = None
        self._item_embeddings = None

    def fit(self, x: Tuple[Sequence, Sequence], y: Sequence) -> None:
        """Build user and item embeddings.

        :param x: values of user_id and item_id.
        :param y: ratings for corresponding user_id and item_id in `x`.
        """
        # Preprocess ratings
        user_ids = pd.Series(x[0], dtype='category')
        item_ids = pd.Series(x[1], dtype='category')
        self._user_categories = CategoricalDtype(user_ids.cat.categories)
        self._item_categories = CategoricalDtype(item_ids.cat.categories)

        # To avoid losing zero ratings when converting into sparse matrix
        ratings = np.array(y, dtype=np.float64).ravel()
        ratings[ratings == 0] = np.finfo(float).eps

        # Build the interaction matrix
        interactions = coo_matrix(
            (ratings, (user_ids.cat.codes, item_ids.cat.codes))).tocsr()

        # Scale ratings
        self._fit_scaler(interactions)
        self._scale_ratings(interactions)

        # Compute svd decomposition
        u, sigma, vt = svds(interactions, k=self.embed_size)

        # Compose user and item embeddings
        sigma_sqrt = np.sqrt(np.eye(self.embed_size) * sigma)
        self._user_embeddings = np.dot(u, sigma_sqrt)
        self._item_embeddings = np.dot(sigma_sqrt, vt)

    def predict(self, x: Tuple[Sequence, Sequence],
                chunk_size: Optional[int] = 500_000,
                progress_bar: bool = False) -> np.ndarray:
        """Calculate predicted rating.

        :param x: values of user_id and item_id.
        :param chunk_size: perform predictions in chunks to reduce memory
        consumption.
        :param progress_bar: if to show progress bar.
        :return: predicted ratings.
        """
        # Preprocess users, items
        user_indices = pd.Series(x[0], dtype=self._user_categories).cat.codes
        item_indices = pd.Series(x[1], dtype=self._item_categories).cat.codes

        # Split in chunks
        samples_count = len(user_indices)
        chunk_size = samples_count if not chunk_size \
            else min(chunk_size, samples_count)
        predictions = np.zeros(samples_count)
        for index in tqdm(range(0, samples_count, chunk_size),
                          disable=not progress_bar):
            # Get chunk of data
            slice_ = slice(index, min(index + chunk_size, samples_count))
            user_indices_chunk = user_indices[slice_]
            item_indices_chunk = item_indices[slice_]

            # Calculate ratings
            predictions[slice_] = np.sum(
                self._user_embeddings[user_indices_chunk, :]
                * self._item_embeddings[:, item_indices_chunk].T,
                axis=1)
        return self._unscale_ratings(predictions, user_indices)
