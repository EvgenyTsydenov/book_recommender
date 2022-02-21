from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class RatingScaleMixin:
    """Provide functionality to scale ratings."""

    def __init__(self, rating_normalization: Optional[str] = 'mean') -> None:
        self.rating_normalization = rating_normalization
        self._scaler = None

    def _fit_scaler(self, interactions: csr_matrix) -> None:
        """Calculate statistics to scale ratings.

        :param interactions: interactions between users and items.
        """
        bias_ = np.zeros(interactions.shape[0])
        scale_ = np.ones(interactions.shape[0])
        count = interactions.getnnz(axis=1)
        if self.rating_normalization == 'mean':
            bias_ = np.divide(interactions.sum(axis=1).A1, count,
                              out=np.zeros(len(count)), where=count != 0)
        elif self.rating_normalization == 'z-score':
            bias_ = np.divide(interactions.sum(axis=1).A1, count,
                              out=np.zeros(len(count)), where=count != 0)
            interactions_ = interactions.copy()
            interactions_.data **= 2
            bias_sq = np.divide(interactions_.sum(axis=1).A1, count,
                                out=np.zeros(len(count)), where=count != 0)
            scale_ = np.sqrt(bias_sq - bias_ ** 2)
            scale_[scale_ == 0] = 1
        elif self.rating_normalization is not None:
            raise ValueError(f'Unknown normalization strategy: '
                             f'{self.rating_normalization}')
        self._scaler = np.column_stack([bias_, scale_])

    def _unscale_ratings(self, ratings: np.ndarray,
                         indices: np.ndarray) -> np.ndarray:
        """Convert ratings to the original scale.

        :param indices: users' or items' indices.
        :param ratings: rating values.
        :return: unscaled ratings.
        """
        return ratings * self._scaler[indices, 1] + self._scaler[indices, 0]

    def _scale_ratings(self, interactions: csr_matrix) -> None:
        """Scale rating values in the interaction matrix.

        Scaling is performed inplace.

        :param interactions: ratings to scale.
        """
        row_indices, col_indices = interactions.nonzero()
        interactions[row_indices, col_indices] -= \
            self._scaler[row_indices, 0]
        interactions[row_indices, col_indices] /= \
            self._scaler[row_indices, 1]
