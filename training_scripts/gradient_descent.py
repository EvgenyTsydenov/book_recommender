import os
import pickle
import time
from typing import Dict, Tuple

import neptune.new as neptune
import optuna
import pandas as pd
import tensorflow as tf
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from sqlalchemy.engine import URL
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

# noinspection PyUnresolvedReferences
import shared
from models.gradient_descent import RecommenderGD
from shared import NEPTUNE_API_KEY, NEPTUNE_PROJECT, RANDOM_SEED
from utils.training import ObjectiveBase


class ObjectiveGD(ObjectiveBase):
    """Objective to minimize for hyperparameter tuning.

    Matrix factorization recommender optimized with gradient descent.
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

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter tuning.

        Training matrix factorization recommender based on the gradient
        descent.

        :param trial: optuna's trial object.
        :return: value of metric to optimize.
        """

        # Define params
        model_param = {
            'embed_size': trial.suggest_int('embed_size', 5, 100),
            'l2_value': trial.suggest_loguniform('l2_value', 1e-15, 1e-2),
        }
        optimize_params = {
            'loss': 'huber',
            'optimizer': 'adam',
            'lr_init': 3e-4,
            'lr_scheduler': 'plateau',
            'lr_reduce_factor': 0.2,
            'lr_reduce_min_delta': 1e-9,
            'lr_reduce_min_lr': 1e-6,
            'lr_reduce_patience': 3
        }
        train_params = {
            'device': 'GPU:0',
            'epoch_count': 100,
            'early_stopping': True,
            'early_stopping_patience': 10,
            'batch_size': 8192
        }

        # Check duplication and skip if it's detected
        for old_trial in trial.study.trials:
            if old_trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if old_trial.params == trial.params:
                raise optuna.TrialPruned()

        # For experiment tracking
        neptune_run = neptune.init(
            project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_KEY, tags=['GD'],
            description='Optuna tunes hyperparameters', capture_stdout=False,
            capture_stderr=False, capture_hardware_metrics=False,
            source_files=['../models/gradient_descent.py',
                          'gradient_descent.py'])

        # Log params
        neptune_run['model'] = model_param
        neptune_run['training'] = train_params
        neptune_run['optimization'] = optimize_params
        neptune_run['random_seed'] = RANDOM_SEED
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
        neptune_run['optuna/study_name'] = trial.study.study_name
        neptune_run['optuna/trial_number'] = trial.number

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
            if optimize_params['lr_scheduler'] == 'plateau':
                lr_sch = ReduceLROnPlateau(
                    factor=optimize_params['lr_reduce_factor'],
                    min_lr=optimize_params['lr_reduce_min_lr'],
                    min_delta=optimize_params['lr_reduce_min_delta'],
                    patience=optimize_params['lr_reduce_patience'])
                callbacks.append(lr_sch)

            with tf.device(f'/{train_params["device"]}'):

                # Create model
                recommender = RecommenderGD(
                    users_count=len(self.user_cats.categories),
                    books_count=len(self.item_cats.categories),
                    embed_size=model_param['embed_size'],
                    l2_regularizer=model_param['l2_value'],
                    random_seed=RANDOM_SEED)

                # Compile
                recommender.compile(
                    loss=self._get_loss_func(optimize_params['loss']),
                    optimizer=self._get_optimizer(
                        optimizer_name=optimize_params['optimizer'],
                        lr_init=optimize_params['lr_init']),
                    metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

                # Train
                start_time = time.time()
                history = recommender.fit(
                    x=[self._user_ids_train, self._item_ids_train],
                    y=self._ratings_train_true,
                    epochs=train_params['epoch_count'],
                    validation_data=([self._user_ids_val, self._item_ids_val],
                                     self._ratings_val_true),
                    batch_size=train_params['batch_size'],
                    callbacks=callbacks)
                end_time = time.time()
                neptune_run['training/time_min'] = \
                    round((end_time - start_time) / 60, 2)

                # Predict
                ratings_train_predict = recommender.predict(
                    [self._user_ids_train, self._item_ids_train],
                    batch_size=8192)
                ratings_val_predict = recommender.predict(
                    (self._user_ids_val, self._item_ids_val),
                    batch_size=8192)

            # Evaluate
            scores = {}
            ratings_train_predict_unscaled = self._unscale(
                self._user_ids_train, ratings_train_predict)
            scores.update(self.get_metrics(self._ratings_train_true,
                                           ratings_train_predict))
            scores.update(self.get_metrics(self._ratings_train_true_unscaled,
                                           ratings_train_predict_unscaled,
                                           'train_unscaled'))
            ratings_val_predict_unscaled = self._unscale(
                self._user_ids_val, ratings_val_predict)
            scores.update(self.get_metrics(self._ratings_val_true,
                                           ratings_val_predict, 'val'))
            scores.update(self.get_metrics(self._ratings_val_true_unscaled,
                                           ratings_val_predict_unscaled,
                                           'val_unscaled'))

            # Log scores amd learning rate
            neptune_run['score'] = scores
            neptune_run['training/lr_epoch'].log(history.history.get('lr', []))

            # Save model
            run_id = neptune_run.get_attribute('sys/id').fetch()
            model_path = os.path.join('../trained_models', f'{run_id}')
            recommender.save(model_path)
            neptune_run['model/path'] = os.path.abspath(model_path)

        finally:
            neptune_run.stop()
        return scores['val_rmse']

    def _get_loss_func(self, loss_name: str) -> tf.keras.losses.Loss:
        """Create loss function.

        :param loss_name: name of loss function
        :return: loss object.
        """
        if loss_name == 'huber':
            return Huber()
        if loss_name == 'mse':
            return MeanSquaredError()
        raise ValueError(f'Unknown name of the loss function — {loss_name}.')

    def _get_optimizer(self, optimizer_name: str, **kwargs) \
            -> tf.keras.optimizers.Optimizer:
        """Create optimizer.

        :param optimizer_name: name of optimizer.
        :return: optimizer instance.
        """
        if optimizer_name == 'adam':
            lr = kwargs.get('lr_init', 0.001)
            return Adam(learning_rate=lr)
        raise ValueError(f'Unknown name of the optimizer — {optimizer_name}.')


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
    objective = ObjectiveGD(work_ratings_train, work_ratings_val, scalers)
    study_name = 'gradient_descent'
    storage_url = URL.create(drivername=os.environ['OPTUNA_DB_DRIVER'],
                             username=os.environ['OPTUNA_DB_USER'],
                             password=os.environ['OPTUNA_DB_PASSWORD'],
                             host=os.environ['OPTUNA_DB_HOST'],
                             port=os.environ['OPTUNA_DB_PORT'],
                             database=os.environ['OPTUNA_DB_NAME'])
    storage = optuna.storages.RDBStorage(url=str(storage_url),
                                         engine_kwargs={'pool_recycle': 3600})
    study_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(sampler=study_sampler, direction='minimize',
                                storage=storage, load_if_exists=True,
                                study_name=study_name)
    study.optimize(objective, n_trials=20)
