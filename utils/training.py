from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from pandas import CategoricalDtype
from sklearn.metrics import mean_squared_error as mse, \
    mean_absolute_error as mae


class ObjectiveBase(ABC):
    """Objective to minimize for hyperparameter tuning."""

    def __init__(self, ratings_train: pd.DataFrame,
                 ratings_val: pd.DataFrame,
                 user_ratings_scalers: Dict[str, Tuple[float, float]]):
        """Create an objective.

        :param ratings_train: user ratings for training.
        :param ratings_val: user ratings for validation.
        :param user_ratings_scalers: standard scaler parameters.
        """

        # For training
        self.user_cats = self._get_categories(ratings_train, 'user_id')
        self.item_cats = self._get_categories(ratings_train, 'work_id')
        self._user_ids_train, self._item_ids_train, self._ratings_train_true \
            = self._prepare_data(ratings_train)
        self._user_ids_val, self._item_ids_val, self._ratings_val_true \
            = self._prepare_data(ratings_val)

        # For evaluation
        self._user_scale, self._user_mean = \
            self._prepare_scalers(user_ratings_scalers)
        self._ratings_train_true_unscaled = self._unscale(
            self._user_ids_train, self._ratings_train_true)
        self._ratings_val_true_unscaled = self._unscale(
            self._user_ids_val, self._ratings_val_true)

    @abstractmethod
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter tuning.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """
        pass

    def _unscale(self, user_ids: np.ndarray,
                 scaled_ratings: np.ndarray) -> np.ndarray:
        """Unscale ratings to the original range.

        :param user_ids: user ids whose ratings are passed.
        :param scaled_ratings: ratings to unscale.
        :return: unscaled ratings.
        """
        return scaled_ratings * self._user_scale[user_ids] \
               + self._user_mean[user_ids]

    def _prepare_scalers(self, scalers: Dict[str, Tuple[float, float]]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Prepare user scalers for fast usage.

        :param scalers: user scalers info.
        :return: mean and standard deviations of each user's ratings.
        """
        scaler_pd = pd.DataFrame.from_dict(
            scalers, orient='index', columns=['scale', 'mean'])
        cats_pd = pd.DataFrame(
            data={'cat_code': range(len(self.user_cats.categories))},
            index=self.user_cats.categories)
        scaler_pd = scaler_pd.join(cats_pd).sort_values('cat_code')
        return np.hstack(scaler_pd['scale']).reshape(-1, 1), \
               np.hstack(scaler_pd['mean']).reshape(-1, 1)

    def _prepare_data(self, data: pd.DataFrame) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert data in the proper types.

        :param data: dataset.
        :return: user indexes, item indexes, rating values.
        """
        data['user_id'] = data['user_id'].astype(self.user_cats)
        data['work_id'] = data['work_id'].astype(self.item_cats)
        data['rating'] = data['rating'].astype('float16')
        data['rating_scaled'] = data['rating_scaled'].astype('float16')

        # Drop unknown users or items
        mask = data['user_id'].isna() | data['work_id'].isna()
        data = data[~mask]

        # Prepare result
        user_ids = data['user_id'].cat.codes.values
        item_ids = data['work_id'].cat.codes.values
        ratings_true = data[['rating_scaled']].values
        return user_ids, item_ids, ratings_true

    def _get_categories(self, ratings_train: pd.DataFrame, column: str) \
            -> CategoricalDtype:
        """Extract user and item categories from training ratings.

        :param ratings_train: user ratings for training.
        :param column: column which values to use.
        :return: categories.
        """
        ratings_train[column] = ratings_train[column].astype('category')
        return CategoricalDtype(
            categories=ratings_train[column].cat.categories)

    def get_metrics(self, y_predict: np.ndarray, y_true: np.ndarray,
                    name: str = 'train') -> Dict[str, float]:
        """Calculate MAE and RMSE for given data.

        :param y_predict: predicted values.
        :param y_true: true values.
        :param name: name of data (val, train, train_unscaled, etc.)
        :return: dict with metrics.
        """
        return {
            f'{name}_rmse': mse(y_true, y_predict, squared=False),
            f'{name}_mae': mae(y_true, y_predict)
        }
