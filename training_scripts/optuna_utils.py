from abc import ABC, abstractmethod
from typing import Optional

import optuna


class OptunaObjective(ABC):
    """Objective to minimize for hyperparameter tuning.

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
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.neptune_project = neptune_project_name
        self.neptune_key = neptune_api_key
        self.random_seed = random_seed
        self.model_path = model_path

    @abstractmethod
    def __call__(self, trial: optuna.trial.Trial) -> float:
        raise NotImplemented

    def _is_duplicated_trial(self, trial: optuna.trial.Trial) -> bool:
        """Check if trial with the same parameters already exists.

        :param trial: optuna trial.
        :return: True if trial with the same parameters already exists.
        """
        for old_trial in trial.study.trials:
            if old_trial.number == trial.number:
                continue
            if old_trial.params == trial.params:
                return True
        return False
