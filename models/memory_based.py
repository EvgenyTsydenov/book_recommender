from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Iterable

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from scipy.sparse import coo_matrix, csr_matrix, vstack, find
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from models.mixins import RatingScaleMixin


class RecommenderMemoryBased(RatingScaleMixin, ABC):
    """Base class for collaborative filtering memory-based recommenders.

    :param min_overlaps: minimal number of overlaps to keep a similarity value.
    If a similarity is based on the number of common ratings
    that is less than `min_overlaps`, the similarity will be set to zero.
    :param overlaps_penalty: value to decrease similarity according
    to the number of overlaps it is based. The similarity will be penalized
    by a factor proportional to the number of overlaps if this number
    is less than a given parameter `overlaps_penalty`. If both `min_overlaps`
    and `overlaps_penalty` are passed, only the latter will be used.
    :param negative_sim_filtering: if to remove negative similarities.
    :param min_sim_threshold: minimal value of similarities to keep.
    Similarities less than this value will be set to zero.
    :param top_n_sim: the number of similarities to keep as neighbors.
    :param similarity_measure: which metric to use for estimation of similarity
    between items or users ("cosine", "cosine_adjusted", etc.)
    :param rating_normalization: which rating normalization strategy
    to use ("mean", "z-score", or None)
    :param rating_weighting: if to use similarity values to weigh ratings.
    """

    def __init__(self, similarity_measure: str = 'cosine_adjusted',
                 rating_normalization: Optional[str] = 'mean',
                 rating_weighting: bool = True,
                 negative_sim_filtering: bool = True,
                 min_sim_threshold: Optional[float] = None,
                 top_n_sim: Optional[int] = None,
                 min_overlaps: Optional[int] = None,
                 overlaps_penalty: Optional[int] = None) -> None:

        # Save arguments
        super().__init__(rating_normalization=rating_normalization)
        self.negative_sim_filtering = negative_sim_filtering
        self.min_sim_threshold = min_sim_threshold
        self.top_n_sim = top_n_sim
        self.min_overlaps = min_overlaps
        self.overlaps_penalty = overlaps_penalty
        self.similarity_measure = similarity_measure
        self.rating_weighting = rating_weighting

        # To convert IDs of users and items into their indices
        self._item_categories = None
        self._user_categories = None

        # Sparse representation of interactions
        self._interactions = None

        # Sparse representation of similarities between users or items
        self._similarities = None

    @abstractmethod
    def _build_interaction_matrix(self, ratings: np.ndarray,
                                  user_indices: np.ndarray,
                                  item_indices: np.ndarray) -> csr_matrix:
        """Build sparse representation of the interaction matrix.

        :param ratings: ratings.
        :param user_indices: user indices.
        :param item_indices: item indices.
        :return: interaction matrix.
        """
        raise NotImplemented

    @abstractmethod
    def _get_centered_interactions(self, interactions: csr_matrix) \
            -> csr_matrix:
        """Adjust interactions with average values of users' ratings.

        :param interactions: interactions to center.
        :return: centered copy of interactions.
        """
        raise NotImplementedError

    @abstractmethod
    def _predict(self, user_indices: np.ndarray,
                 item_indices: np.ndarray) -> np.ndarray:
        """Calculate predictions.

        :param user_indices: indices of users whose ratings to predict.
        :param item_indices: indices of items whose ratings to predict.
        :return: predicted ratings.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _sim_categories(self) -> Iterable:
        """Entries of the similarity matrix"""
        raise NotImplementedError

    def fit(self, x: Tuple[Sequence, Sequence], y: Sequence,
            chunk_size: Optional[int] = None,
            progress_bar: bool = False) -> None:
        """Calculate similarities for memory-based based recommender.

        This also scales the interaction matrix and converts it into the proper
         format for faster predictions.

        :param x: values of user_id and item_id.
        :param y: ratings for corresponding user_id and item_id in `x`.
        :param chunk_size: size of a chunk to calculate similarities in chunks.
        :param progress_bar: if to show progress bar when calculating
        similarities.
        """
        # Preprocess users and items
        user_ids = pd.Series(x[0], dtype='category')
        item_ids = pd.Series(x[1], dtype='category')
        self._user_categories = CategoricalDtype(user_ids.cat.categories)
        self._item_categories = CategoricalDtype(item_ids.cat.categories)

        # To avoid losing zero ratings when converting into sparse matrix
        ratings = np.array(y, dtype=np.float64).ravel()
        ratings[ratings == 0] = np.finfo(float).eps

        # Build the interaction matrix
        interactions = self._build_interaction_matrix(
            ratings, user_ids.cat.codes, item_ids.cat.codes)

        # Calculate similarities
        self._similarities = self._calculate_similarities(
            interactions, chunk_size, progress_bar)

        # Calculate statistics and scale ratings
        # according to `rating_normalization` strategy
        self._fit_scaler(interactions)
        self._scale_ratings(interactions)

        # Transform the interaction matrix for faster predictions
        self._interactions = interactions.T.tocsr()

    def predict(self, x: Tuple[Sequence, Sequence],
                chunk_size: Optional[int] = 50_000,
                progress_bar: bool = False) -> np.ndarray:
        """Calculate predictions.

        :param x: values of user_id and item_id.
        :param chunk_size: size of a chunk to calculate in chunks. Since index
        pointer of sparse matrices is a value of type np.int32, it can be
        overflowed when too many samples are passed to predict. In this case,
        it is necessary to forcibly split the samples into chunks. Although
        it is possible to calculate exact value of max chunk size to avoid
        this overflow, a static value is preset here for simplicity.
        :param progress_bar: if to show progress bar.
        :return: predicted values.
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

            # Calculate predictions
            predictions[slice_] = self._predict(user_indices_chunk,
                                                item_indices_chunk)
        return predictions

    @property
    def similarities(self) -> pd.DataFrame:
        """Similarities represented as a dataframe."""
        mapper = dict(enumerate(self._sim_categories))
        sources, targets, values = find(self._similarities)
        return pd.DataFrame({'id_source': pd.Series(sources).map(mapper),
                             'id_similar': pd.Series(targets).map(mapper),
                             'similarity': values})

    def save_similarities(self, path_to_save: str,
                          progress_bar: bool = False) -> None:
        """Save similarities to csv file.

        :param path_to_save: path where to save.
        :param progress_bar: if to show progress bar.
        """
        mapper = dict(enumerate(self._sim_categories))
        sources, targets, values = find(self._similarities)
        header = f'id_source,id_similar,similarity\n'
        with open(path_to_save, 'w') as file:
            file.write(header)
            for source, target, value in tqdm(zip(sources, targets, values),
                                              total=len(sources),
                                              disable=not progress_bar):
                file.write(f'{mapper[source]},{mapper[target]},{value}\n')

    def _calculate_similarities(self, interactions: csr_matrix,
                                chunk_size: Optional[int] = None,
                                progress_bar: bool = False) -> csr_matrix:
        """Calculate similarities.

        :param interactions: interaction matrix to calculate similarities.
        :param chunk_size: size of a chunk to calculate in chunks.
        :param progress_bar: if to show progress bar.
        :return: similarities.
        """
        # Calculate similarity with specified `similarity_measure`
        if self.similarity_measure == 'cosine':
            return self._get_cos_similarities(interactions, chunk_size,
                                              progress_bar)
        if self.similarity_measure == 'cosine_adjusted':
            interactions_ = self._get_centered_interactions(interactions)
            return self._get_cos_similarities(interactions_, chunk_size,
                                              progress_bar)
        raise ValueError(f'Unknown similarity measure: '
                         f'{self.similarity_measure}')

    def _get_cos_similarities(self, interactions: csr_matrix,
                              chunk_size: Optional[int] = None,
                              progress_bar: bool = False) -> csr_matrix:
        """Calculate cosine similarity between rows of the interaction matrix.

        :param interactions: interaction matrix.
        :param chunk_size: size of a chunk to calculate in chunks.
        :param progress_bar: if to show progress bar.
        :return: similarity matrix.
        """
        # To calculate overlaps
        need_overlaps = self.min_overlaps or self.overlaps_penalty
        is_rated = None
        if need_overlaps:
            is_rated = interactions.copy()
            is_rated.data.fill(1.0)

        # Split in chunks
        chunks = []
        rows_count = interactions.shape[0]
        chunk_size = rows_count if chunk_size is None \
            else min(chunk_size, rows_count)
        for index in tqdm(range(0, rows_count, chunk_size),
                          disable=not progress_bar):
            # Indices of rows to handle in chunk
            slice_ = slice(index, min(index + chunk_size, rows_count))

            # Get slice of interactions
            inter_chunk = interactions[slice_]

            # Compute similarity
            sim_chunk = cosine_similarity(inter_chunk, interactions,
                                          dense_output=False)

            # Since each item or user has the highest similarity with itself,
            # set diagonal values to zero
            self._drop_diagonal(sim_chunk, slice_.start)

            # Correct similarities according to the number of overlaps
            if need_overlaps:
                overlaps = is_rated[slice_].dot(is_rated.T)
                sim_chunk = self._adjust_similarities(sim_chunk, overlaps)

            # Filter similarities
            self._threshold_sim_filter(sim_chunk)
            self._top_n_sim_filter(sim_chunk)

            # Save result
            chunks.append(sim_chunk)

        # Concat result of chunks and return as csr matrix
        return vstack(chunks, format='csr')

    def _adjust_similarities(self, similarity: csr_matrix,
                             overlaps: csr_matrix) -> csr_matrix:

        """Correct values of similarities according to the number of overlaps.

        :param similarity: similarity matrix.
        :param overlaps: number of overlaps between interactions.
        :return: penalized similarity matrix.
        """
        if self.overlaps_penalty:
            penalty = overlaps.copy()
            penalty.data /= self.overlaps_penalty
            penalty.data[penalty.data > 1] = 1
        else:
            penalty = overlaps >= self.min_overlaps
        similarity_adjusted = similarity.multiply(penalty)
        similarity_adjusted.eliminate_zeros()
        return similarity_adjusted

    def _drop_diagonal(self, similarity: csr_matrix,
                       offset: Optional[int] = 0) -> None:
        """Set diagonal elements of similarity matrix to zero.

        The similarity matrix is changed inplace.

        :param similarity: similarity matrix.
        :param offset: number of upper diagonal.
        """
        similarity.setdiag(0, offset)
        similarity.eliminate_zeros()

    def _top_n_sim_filter(self, similarity: csr_matrix) -> None:
        """Keep only N of the highest similarities per row.

        The similarity matrix is changed inplace.

        :param similarity: similarity matrix.
        """
        if self.top_n_sim:

            # Unfortunately, more efficient solution
            # for sparse matrices was not found
            for i in range(0, similarity.shape[0]):

                # Get the row slice
                row_array = similarity.data[similarity.indptr[i]
                                            :similarity.indptr[i + 1]]

                # If row contains more elements than top_n_sim
                if row_array.shape[0] > self.top_n_sim:
                    indices = np.argpartition(row_array, -self.top_n_sim)
                    row_array[indices[:-self.top_n_sim]] = 0
            similarity.eliminate_zeros()

    def _threshold_sim_filter(self, similarity: csr_matrix) -> None:
        """Apply threshold and negative filtering strategies.

        The similarity matrix is changed inplace.

        :param similarity: similarity matrix.
        """
        sim_threshold = 0. if self.negative_sim_filtering else None
        if self.min_sim_threshold is not None:
            if sim_threshold is not None:
                sim_threshold = max(sim_threshold, self.min_sim_threshold)
            else:
                sim_threshold = self.min_sim_threshold
        if sim_threshold is not None:
            similarity.data[similarity.data < sim_threshold] = 0
            similarity.eliminate_zeros()


class RecommenderII(RecommenderMemoryBased):
    """Collaborative filtering item-based recommender.

    :param min_overlaps: minimal number of overlaps to keep a similarity value.
    If a similarity is based on the number of common ratings
    that is less than `min_overlaps`, the similarity will be set to zero.
    :param overlaps_penalty: value to decrease similarity according
    to the number of overlaps it is based. The similarity will be penalized
    by a factor proportional to the number of overlaps if this number
    is less than a given parameter `overlaps_penalty`. If both `min_overlaps`
    and `overlaps_penalty` are passed, only the latter will be used.
    :param negative_sim_filtering: if to remove negative similarities.
    :param min_sim_threshold: minimal value of similarities to keep.
    Similarities less than this value will be set to zero.
    :param top_n_sim: the number of similarities to keep as neighbors.
    :param similarity_measure: which metric to use for estimation of similarity
    between items or users ("cosine", "cosine_adjusted", etc.)
    :param rating_normalization: which rating normalization strategy
    to use ("mean", "z-score", or None)
    :param rating_weighting: if to use similarity values to weigh ratings.
    """

    def __init__(self, similarity_measure: str = 'cosine_adjusted',
                 rating_normalization: Optional[str] = 'mean',
                 rating_weighting: bool = True,
                 negative_sim_filtering: bool = True,
                 min_sim_threshold: Optional[float] = None,
                 top_n_sim: Optional[int] = None,
                 min_overlaps: Optional[int] = None,
                 overlaps_penalty: Optional[int] = None) -> None:
        super().__init__(similarity_measure, rating_normalization,
                         rating_weighting, negative_sim_filtering,
                         min_sim_threshold, top_n_sim, min_overlaps,
                         overlaps_penalty)

    def _build_interaction_matrix(self, ratings: np.ndarray,
                                  user_indices: np.ndarray,
                                  item_indices: np.ndarray) -> csr_matrix:
        """Build sparse representation of the interaction matrix.

        :param ratings: ratings.
        :param user_indices: user indices.
        :param item_indices: item indices.
        :return: interaction matrix.
        """
        # Transpose interaction matrix for item-based recommender
        return coo_matrix((ratings, (item_indices, user_indices))).tocsr()

    def _get_centered_interactions(self, interactions: csr_matrix) \
            -> csr_matrix:
        """Adjust interactions with average values of users' ratings.

        :param interactions: interactions in form (items, users).
        :return: centered copy of interactions.
        """
        # Find non-zero values
        item_indices, user_indices = interactions.nonzero()

        # Calculate mean rating per user
        count = interactions.getnnz(axis=0)
        user_mean = np.divide(interactions.sum(axis=0).A1, count,
                              out=np.zeros(len(count)), where=count != 0)

        # Have to copy original matrix to avoid its modification
        interactions_ = interactions.copy()
        interactions_[item_indices, user_indices] -= user_mean[user_indices]
        return interactions_

    def _predict(self, user_indices: np.ndarray,
                 item_indices: np.ndarray) -> np.ndarray:
        """Calculate predictions.

        :param user_indices: indices of users whose ratings to predict.
        :param item_indices: indices of items whose ratings to predict.
        :return: predicted ratings.
        """
        # Get existing ratings and similarities
        item_sim = self._similarities[item_indices]
        user_ratings = self._interactions[user_indices]

        # Get numerator and denominator to calculate ratings
        if self.rating_weighting:
            ratings = item_sim.multiply(user_ratings).sum(axis=1)
            item_sim.data = np.abs(item_sim.data)
        else:
            item_sim.data.fill(1.0)
            ratings = item_sim.multiply(user_ratings).sum(axis=1)
        user_ratings.data.fill(1.0)
        scale = item_sim.multiply(user_ratings).sum(axis=1)

        # If common ratings and similarities were not found,
        # predict zero that means average ratings of item
        predictions = np.divide(ratings, scale, where=scale != 0,
                                out=np.zeros_like(ratings)).A1
        return self._unscale_ratings(predictions, item_indices)

    @property
    def _sim_categories(self) -> pd.Index:
        """Entries of the similarity matrix"""
        return self._item_categories.categories


class RecommenderUU(RecommenderMemoryBased):
    """Collaborative filtering user-based recommender.

    :param min_overlaps: minimal number of overlaps to keep a similarity value.
    If a similarity is based on the number of common ratings
    that is less than `min_overlaps`, the similarity will be set to zero.
    :param overlaps_penalty: value to decrease similarity according
    to the number of overlaps it is based. The similarity will be penalized
    by a factor proportional to the number of overlaps if this number
    is less than a given parameter `overlaps_penalty`. If both `min_overlaps`
    and `overlaps_penalty` are passed, only the latter will be used.
    :param negative_sim_filtering: if to remove negative similarities.
    :param min_sim_threshold: minimal value of similarities to keep.
    Similarities less than this value will be set to zero.
    :param top_n_sim: the number of similarities to keep as neighbors.
    :param similarity_measure: which metric to use for estimation of similarity
    between items or users ("cosine", "cosine_adjusted", etc.)
    :param rating_normalization: which rating normalization strategy
    to use ("mean", "z-score", or None)
    :param rating_weighting: if to use similarity values to weigh ratings.
    """

    def __init__(self, similarity_measure: str = 'cosine_adjusted',
                 rating_normalization: Optional[str] = 'mean',
                 rating_weighting: bool = True,
                 negative_sim_filtering: bool = True,
                 min_sim_threshold: Optional[float] = None,
                 top_n_sim: Optional[int] = None,
                 min_overlaps: Optional[int] = None,
                 overlaps_penalty: Optional[int] = None) -> None:
        super().__init__(similarity_measure, rating_normalization,
                         rating_weighting, negative_sim_filtering,
                         min_sim_threshold, top_n_sim, min_overlaps,
                         overlaps_penalty)

    def _build_interaction_matrix(self, ratings: np.ndarray,
                                  user_indices: np.ndarray,
                                  item_indices: np.ndarray) -> csr_matrix:
        """Build sparse representation of the interaction matrix.

        :param ratings: ratings.
        :param user_indices: user indices.
        :param item_indices: item indices.
        :return: interaction matrix.
        """
        # Save interaction matrix for user-based recommender as it is
        return coo_matrix((ratings, (user_indices, item_indices))).tocsr()

    def _get_centered_interactions(self, interactions: csr_matrix) \
            -> csr_matrix:
        """Adjust interactions with average values of users' ratings.

        :param interactions: interactions in form (users, items).
        :return: centered copy of interactions.
        """
        # Find non-zero values
        user_indices, item_indices = interactions.nonzero()

        # Calculate mean rating per user
        count = interactions.getnnz(axis=1)
        user_mean = np.divide(interactions.sum(axis=1).A1, count,
                              out=np.zeros(len(count)), where=count != 0)

        # Have to copy original matrix to avoid its modification
        interactions = interactions.copy()
        interactions[user_indices, item_indices] -= user_mean[user_indices]
        return interactions

    def _predict(self, user_indices: np.ndarray,
                 item_indices: np.ndarray) -> np.ndarray:
        """Calculate predictions.

        :param user_indices: indices of users whose ratings to predict.
        :param item_indices: indices of items whose ratings to predict.
        :return: predicted ratings.
        """
        # Get existing ratings and similarities
        user_sim = self._similarities[user_indices]
        item_ratings = self._interactions[item_indices]

        # Get numerator and denominator to calculate ratings
        if self.rating_weighting:
            ratings = user_sim.multiply(item_ratings).sum(axis=1)
            user_sim.data = np.abs(user_sim.data)
        else:
            user_sim.data.fill(1.0)
            ratings = user_sim.multiply(item_ratings).sum(axis=1)
        item_ratings.data.fill(1.0)
        scale = user_sim.multiply(item_ratings).sum(axis=1)

        # If common ratings and similarities were not found,
        # predict zero that means average ratings of user
        predictions = np.divide(ratings, scale, where=scale != 0,
                                out=np.zeros_like(scale)).A1
        return self._unscale_ratings(predictions, user_indices)

    @property
    def _sim_categories(self) -> pd.Index:
        """Entries of the similarity matrix"""
        return self._user_categories.categories
