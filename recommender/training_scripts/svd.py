import os
import pickle
import time
from typing import Dict, Tuple

import neptune.new as neptune
import numpy as np
import optuna
import pandas as pd
import scipy.sparse
from pandas import CategoricalDtype
from scipy.sparse import find, coo_matrix
from sklearn.metrics import mean_squared_error as mse, \
    mean_absolute_error as mae

# noinspection PyUnresolvedReferences
import shared
from recommender.models.svd import RecommenderSVD
from shared import NEPTUNE_API_KEY, NEPTUNE_PROJECT, RANDOM_SEED


class ObjectiveSVD:
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
        self._user_cats = self._get_categories(ratings_train, 'user_id')
        self._item_cats = self._get_categories(ratings_train, 'work_id')
        self._ratings_train = self._prepare_data(ratings_train)
        self._ratings_val = self._prepare_data(ratings_val)
        self._interacts_train = self._get_interactions(self._ratings_train)
        self._interacts_val = self._get_interactions(self._ratings_val)

        # For evaluation
        self._user_ids_train, self._item_ids_train, self._ratings_train_true \
            = find(self._interacts_train)
        self._user_ids_val, self._item_ids_val, self._ratings_val_true \
            = find(self._interacts_val)
        self._user_scale, self._user_mean = \
            self._prepare_scalers(user_ratings_scalers)
        self._ratings_train_true_unscaled = self._unscale(
            self._user_ids_train, self._ratings_train_true)
        self._ratings_val_true_unscaled = self._unscale(
            self._user_ids_val, self._ratings_val_true)

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
            data={'cat_code': range(len(self._user_cats.categories))},
            index=self._user_cats.categories)
        scaler_pd = scaler_pd.join(cats_pd).sort_values('cat_code')
        return np.hstack(scaler_pd['scale']), np.hstack(scaler_pd['mean'])

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data in the proper types.

        :param data: dataset.
        :return: preprocessed data.
        """
        data['user_id'] = data['user_id'].astype(self._user_cats)
        data['work_id'] = data['work_id'].astype(self._item_cats)
        data['rating'] = data['rating'].astype('float16')
        data['rating_scaled'] = data['rating_scaled'].astype('float16')

        # Drop unknown users or items
        mask = data['user_id'].isna() | data['work_id'].isna()
        data = data[~mask]
        return data

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

    def _get_interactions(self, ratings: pd.DataFrame) \
            -> scipy.sparse.coo_matrix:
        """Build interaction matrix for given ratings.

        :param ratings: ratings.
        :return: sparse interaction matrix.
        """
        return coo_matrix((ratings['rating_scaled'],
                           (ratings['user_id'].cat.codes,
                            ratings['work_id'].cat.codes)))

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for SVD hyperparameter tuning.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """
        # For experiment tracking
        neptune_run = neptune.init(project=NEPTUNE_PROJECT,
                                   api_token=NEPTUNE_API_KEY, tags=['SVD'],
                                   source_files=['../models/svd.py', 'svd.py'],
                                   description='Optuna tunes hyperparameters')

        # Create model
        model_param = {
            'embed_size': trial.suggest_int('embed_size', 5, 3000)
        }
        neptune_run['model'] = model_param
        neptune_run['random_seed'] = RANDOM_SEED
        svd = RecommenderSVD(RANDOM_SEED)

        try:
            # Train
            start_time = time.time()
            svd.fit(self._interacts_train, model_param['embed_size'])
            end_time = time.time()
            neptune_run['training/time_min'] = \
                round((end_time - start_time) / 60, 2)

            # Evaluate with train data
            scores = {}
            ratings_train_predict = svd.predict((self._user_ids_train,
                                                 self._item_ids_train))
            ratings_train_predict_unscaled = self._unscale(
                self._user_ids_train, ratings_train_predict)
            scores.update(self.get_metrics(self._ratings_train_true,
                                           ratings_train_predict))
            scores.update(self.get_metrics(self._ratings_train_true_unscaled,
                                           ratings_train_predict_unscaled,
                                           'train_unscaled'))

            # Evaluate with validation data
            ratings_val_predict = svd.predict((self._user_ids_val,
                                               self._item_ids_val))
            ratings_val_predict_unscaled = self._unscale(self._user_ids_val,
                                                         ratings_val_predict)
            scores.update(self.get_metrics(self._ratings_val_true,
                                           ratings_val_predict, 'val'))
            scores.update(self.get_metrics(self._ratings_val_true_unscaled,
                                           ratings_val_predict_unscaled,
                                           'val_unscaled'))
            neptune_run['score'] = scores
        finally:
            neptune_run.stop()
        return scores['val_rmse']

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


if __name__ == '__main__':
    # Load data
    work_ratings_train = pd.read_csv(
        os.path.join('..', 'data', 'work_ratings_train.csv'))
    work_ratings_val = pd.read_csv(
        os.path.join('..', 'data', 'work_ratings_val.csv'))
    path_scaler = os.path.join('..', 'data', 'user_ratings_scalers.pkl')
    with open(path_scaler, 'rb') as file:
        scalers = pickle.load(file)

    # Start tuning
    objective = ObjectiveSVD(work_ratings_train, work_ratings_val, scalers)
    study_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(sampler=study_sampler, direction="minimize")
    study.optimize(objective, n_trials=15)
