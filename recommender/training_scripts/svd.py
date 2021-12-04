import os
import pickle
import time
from typing import Dict, Tuple

import neptune.new as neptune
import optuna
import pandas as pd
from scipy.sparse import coo_matrix

# noinspection PyUnresolvedReferences
import shared
from recommender.models.svd import RecommenderSVD
from shared import NEPTUNE_API_KEY, NEPTUNE_PROJECT, RANDOM_SEED
from shared.utils import ObjectiveBase


class ObjectiveSVD(ObjectiveBase):
    """Objective to minimize for hyperparameter tuning

    Matrix factorization recommender based on SVD.
    """

    def __init__(self, ratings_train: pd.DataFrame,
                 ratings_val: pd.DataFrame,
                 user_ratings_scalers: Dict[str, Tuple[float, float]]):
        """Create an objective.

        :param ratings_train: user ratings for training.
        :param ratings_val: user ratings for validation.
        :param user_ratings_scalers: standard scaler parameters.
        """
        # Call the base class
        super().__init__(ratings_train, ratings_val, user_ratings_scalers)

        # For training
        self._interacts_train = coo_matrix((self._ratings_train_true,
                                            (self._user_ids_train,
                                             self._item_ids_train)))
        self._interacts_val = coo_matrix((self._ratings_val_true,
                                          (self._user_ids_val,
                                           self._item_ids_val)))

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter tuning.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """

        # Define params
        model_param = {
            'embed_size': trial.suggest_int('embed_size', 5, 1000, log=True)
        }

        # Check duplication and skip if it's detected
        for old_trial in trial.study.trials:
            if old_trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if old_trial.params == trial.params:
                raise optuna.TrialPruned()

        # For experiment tracking
        neptune_run = neptune.init(project=NEPTUNE_PROJECT,
                                   api_token=NEPTUNE_API_KEY, tags=['SVD'],
                                   source_files=['../models/svd.py', 'svd.py'],
                                   description='Optuna tunes hyperparameters',
                                   capture_stdout=False, capture_stderr=False,
                                   capture_hardware_metrics=False)

        # Log model params
        neptune_run['model'] = model_param
        neptune_run['random_seed'] = RANDOM_SEED

        # Log data params
        neptune_run['data/work_ratings_train.csv'] \
            .track_files('../data/work_ratings_train.csv')
        neptune_run['data/work_ratings_val.csv'] \
            .track_files('../data/work_ratings_val.csv')
        neptune_run['data/samples_count_train'] = \
            self._user_ids_train.shape[0]
        neptune_run['data/samples_count_val'] = \
            self._user_ids_val.shape[0]
        neptune_run['data/users_count'] = len(self.user_cats.categories)
        neptune_run['data/books_count'] = len(self.item_cats.categories)

        try:
            # Create model
            svd = RecommenderSVD(RANDOM_SEED)

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
            ratings_val_predict_unscaled = self._unscale(
                self._user_ids_val, ratings_val_predict)
            scores.update(self.get_metrics(self._ratings_val_true,
                                           ratings_val_predict, 'val'))
            scores.update(self.get_metrics(self._ratings_val_true_unscaled,
                                           ratings_val_predict_unscaled,
                                           'val_unscaled'))

            # Log scores
            neptune_run['score'] = scores
        finally:
            neptune_run.stop()
        return scores['val_rmse']


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
    study = optuna.create_study(sampler=study_sampler, direction='minimize')
    study.optimize(objective, n_trials=20)
