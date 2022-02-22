import os
import pickle
import time
from typing import Optional

import neptune.new as neptune
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error as mse, \
    mean_absolute_error as mae
from sqlalchemy.engine import URL

from models.svd import RecommenderSVD
from optuna_utils import OptunaObjective


class ObjectiveSVD(OptunaObjective):
    """Objective to minimize for hyperparameter tuning.

    This tunes collaborative filtering recommender that uses matrix
    factorization approach based on Singular Value Decomposition.

    :param train_data_path: path to data for training.
    :param val_data_path: path to data for validation.
    :param neptune_project_name: name of project in neptune.ai.
    :param neptune_api_key: api key to access neptune.ai.
    :param model_path: folder path to store built model.
    :param random_seed: random seed.
    """

    def __init__(self, train_data_path: str, val_data_path: str,
                 neptune_project_name: str, neptune_api_key: str,
                 random_seed: Optional[int] = None,
                 model_path: Optional[str] = None) -> None:
        super().__init__(train_data_path, val_data_path, neptune_project_name,
                         neptune_api_key, random_seed, model_path)
        self._train_data = pd.read_csv(self.train_data_path)
        self._val_data = pd.read_csv(self.val_data_path)
        self._source_files = ['../models/svd.py', '../models/mixins.py',
                              'svd.py']
        self._tags = ['model-based', 'svd', 'collaborative filtering']
        self._description = 'Optuna tunes collaborative filtering ' \
                            'recommender that uses matrix factorization ' \
                            'approach based on Singular Value Decomposition.'
        self._name = 'SVD-based matrix factorization recommender'

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter tuning.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """
        # Define params
        model_param = {
            'embed_size': trial.suggest_int('embed_size', 5, 100),
            'rating_normalization': trial.suggest_categorical(
                'rating_normalization', ['mean', 'z-score'])
        }

        # If such trial already exists, prune it
        if self._is_duplicated_trial(trial):
            raise optuna.TrialPruned

        # For experiment tracking
        neptune_run = neptune.init(
            project=self.neptune_project, api_token=self.neptune_key,
            tags=self._tags, description=self._description, name=self._name,
            source_files=self._source_files, capture_stdout=False,
            capture_stderr=False, capture_hardware_metrics=False)

        # Log params and data info
        neptune_run['model'] = model_param
        neptune_run['optuna/study_name'] = trial.study.study_name
        neptune_run['optuna/trial_number'] = trial.number
        neptune_run['data/train_data'].track_files(self.train_data_path)
        neptune_run['data/val_data'].track_files(self.val_data_path)
        neptune_run['data/train_samples_count'] = len(self._train_data)
        neptune_run['data/val_samples_count'] = len(self._val_data)
        neptune_run['data/users_count'] = \
            len(self._train_data['user_id'].unique())
        neptune_run['data/works_count'] = \
            len(self._train_data['work_id'].unique())

        try:
            # Create model
            svd = RecommenderSVD(**model_param)

            # Train
            start_time = time.time()
            svd.fit(x=(self._train_data['user_id'],
                       self._train_data['work_id']),
                    y=self._train_data['rating'])
            end_time = time.time()
            neptune_run['training/time_minutes'] = \
                round((end_time - start_time) / 60, 2)

            # Evaluate
            train_predict = svd.predict((self._train_data['user_id'],
                                         self._train_data['work_id']))
            val_predict = svd.predict((self._val_data['user_id'],
                                       self._val_data['work_id']))
            score = {
                'train_rmse': mse(self._train_data['rating'],
                                  train_predict, squared=False),
                'train_mae': mae(self._train_data['rating'], train_predict),
                'val_rmse': mse(self._val_data['rating'],
                                val_predict, squared=False),
                'val_mae': mae(self._val_data['rating'], val_predict)
            }
            neptune_run['score'] = score

            # Save model
            if self.model_path:
                run_id = neptune_run.get_attribute('sys/id').fetch()
                model_path = os.path.join(self.model_path, f'{run_id}.pkl')
                with open(model_path, 'wb') as model_file:
                    pickle.dump(svd, model_file)
                neptune_run['model/path'] = os.path.abspath(model_path)
        finally:
            neptune_run.stop()
        return score['val_rmse']


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Define params
    study_name = 'svd'
    trials_count = 15

    # If the script is executed in parallel,
    # we need to change this per each process to avoid duplicated trials
    optuna_random_seed = int(os.environ['RANDOM_SEED'])

    # Create objective and start tuning
    objective = ObjectiveSVD(
        train_data_path=os.path.join('..', 'data', 'work_ratings_train.csv'),
        val_data_path=os.path.join('..', 'data', 'work_ratings_val.csv'),
        neptune_project_name=os.environ['NEPTUNE_PROJECT'],
        neptune_api_key=os.environ['NEPTUNE_API_KEY'],
        random_seed=int(os.environ['RANDOM_SEED']),
        model_path=os.path.join('..', 'trained_models'))
    storage_url = URL.create(drivername=os.environ['OPTUNA_DB_DRIVER'],
                             username=os.environ['OPTUNA_DB_USER'],
                             password=os.environ['OPTUNA_DB_PASSWORD'],
                             host=os.environ['OPTUNA_DB_HOST'],
                             port=os.environ['OPTUNA_DB_PORT'],
                             database=os.environ['OPTUNA_DB_NAME'])
    storage = optuna.storages.RDBStorage(url=str(storage_url),
                                         engine_kwargs={'pool_recycle': 3600})
    study_sampler = optuna.samplers.TPESampler(seed=optuna_random_seed)
    study = optuna.create_study(sampler=study_sampler, direction='minimize',
                                storage=storage, load_if_exists=True,
                                study_name=study_name)
    study.optimize(objective, n_trials=trials_count)
