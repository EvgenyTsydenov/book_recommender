import os
import time
from typing import Optional

import neptune.new as neptune
import optuna
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from sqlalchemy.engine import URL
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from models.gradient_descent import RecommenderGD
from optuna_utils import OptunaObjective


class ObjectiveGD(OptunaObjective):
    """Objective to minimize for hyperparameter tuning.

    This tunes collaborative filtering recommender that uses matrix
    factorization approach optimized with gradient descent.

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
        dtypes = {'user_id': str, 'work_id': str}
        self._train_data = pd.read_csv(self.train_data_path, dtype=dtypes)
        self._val_data = pd.read_csv(self.val_data_path, dtype=dtypes)
        self._source_files = ['../models/gradient_descent.py',
                              'gradient_descent.py']
        self._tags = ['model-based', 'gradient descent',
                      'collaborative filtering']
        self._description = 'Optuna tunes collaborative filtering ' \
                            'recommender that uses matrix factorization ' \
                            'approach optimized with gradient descent.'
        self._name = 'GD-optimized matrix factorization recommender'

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter tuning.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """
        # Define params
        model_param = {
            'embed_size': trial.suggest_int('embed_size', 5, 100),
            'l2_penalty': trial.suggest_loguniform('l2_penalty', 1e-15, 1e-1),
            'random_seed': self.random_seed
        }
        train_params = {
            'epoch_count': 100,
            'early_stopping': True,
            'early_stopping_patience': 10,
            'batch_size': 8192,
            'loss': 'huber',
            'optimizer': 'adam',
            'lr_init': 3e-4,
            'lr_scheduler': 'plateau',
            'lr_reduce_factor': 0.2,
            'lr_reduce_min_delta': 1e-4,
            'lr_reduce_min_lr': 1e-6,
            'lr_reduce_patience': 3
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
        neptune_run['training'] = train_params
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

            # Clear the global Keras state
            tf.keras.backend.clear_session()

            # Callbacks
            callbacks = [NeptuneCallback(run=neptune_run,
                                         base_namespace='score')]
            if train_params['early_stopping']:
                er_stop = EarlyStopping(
                    patience=train_params['early_stopping_patience'],
                    restore_best_weights=True)
                callbacks.append(er_stop)
            if train_params['lr_scheduler'] == 'plateau':
                lr_sch = ReduceLROnPlateau(
                    factor=train_params['lr_reduce_factor'],
                    min_lr=train_params['lr_reduce_min_lr'],
                    min_delta=train_params['lr_reduce_min_delta'],
                    patience=train_params['lr_reduce_patience'])
                callbacks.append(lr_sch)

            # Create model
            model = RecommenderGD(users=self._train_data['user_id'].unique(),
                                  items=self._train_data['work_id'].unique(),
                                  **model_param)

            # Compile
            optimizer = self._get_optimizer(name=train_params['optimizer'],
                                            lr_init=train_params['lr_init'])
            loss = self._get_loss(train_params['loss'])
            model.compile(loss=loss, optimizer=optimizer,
                          metrics=[RootMeanSquaredError(),
                                   MeanAbsoluteError()])

            # Train
            start_time = time.time()
            history = model.fit(
                x=(self._train_data['user_id'], self._train_data['work_id']),
                y=self._train_data['rating'], callbacks=callbacks,
                epochs=train_params['epoch_count'],
                validation_data=((self._val_data['user_id'],
                                  self._val_data['work_id']),
                                 self._val_data['rating']),
                batch_size=train_params['batch_size'])
            end_time = time.time()
            neptune_run['training/time_minutes'] = \
                round((end_time - start_time) / 60, 2)

            # Evaluate
            train_score = model.evaluate(
                x=(self._train_data['user_id'], self._train_data['work_id']),
                y=self._train_data['rating'],
                batch_size=train_params['batch_size'], return_dict=True)
            val_score = model.evaluate(
                x=(self._val_data['user_id'], self._val_data['work_id']),
                y=self._val_data['rating'],
                batch_size=train_params['batch_size'], return_dict=True)
            neptune_run['score'] = {
                'train_rmse': train_score['root_mean_squared_error'],
                'train_mae': train_score['mean_absolute_error'],
                'val_rmse': val_score['root_mean_squared_error'],
                'val_mae': val_score['mean_absolute_error'],
            }

            # Log learning rate
            neptune_run['training/lr_epoch'].log(history.history.get('lr', []))

            # Save model
            if self.model_path:
                run_id = neptune_run.get_attribute('sys/id').fetch()
                model_path = os.path.join(self.model_path, f'{run_id}')
                model.save(model_path)
                neptune_run['model/path'] = os.path.abspath(model_path)
        finally:
            neptune_run.stop()
        return val_score['root_mean_squared_error']

    def _get_loss(self, name: str, **kwargs) -> tf.keras.losses.Loss:
        """Create loss function.

        :param name: name of loss function
        :return: loss object.
        """
        if name == 'huber':
            return Huber(**kwargs)
        if name == 'mse':
            return MeanSquaredError(**kwargs)
        raise ValueError(f'Unknown name of the loss function — {name}.')

    def _get_optimizer(self, name: str, **kwargs) \
            -> tf.keras.optimizers.Optimizer:
        """Create optimizer.

        :param name: name of optimizer.
        :return: optimizer instance.
        """
        if name == 'adam':
            lr = kwargs.get('lr_init', 0.001)
            return Adam(learning_rate=lr)
        raise ValueError(f'Unknown name of the optimizer — {name}.')


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Define params
    study_name = 'gradient_descent'
    trials_count = 15

    # If the script is executed in parallel,
    # we need to change this per each process to avoid duplicated trials
    optuna_random_seed = int(os.environ['RANDOM_SEED'])

    # Start tuning
    objective = ObjectiveGD(
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
