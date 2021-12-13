import os

import optuna
from sqlalchemy.engine import URL

# noinspection PyUnresolvedReferences
import shared

if __name__ == '__main__':
    study_name = 'gradient_descent'
    storage_url = URL.create(drivername=os.environ['OPTUNA_DB_DRIVER'],
                             username=os.environ['OPTUNA_DB_USER'],
                             password=os.environ['OPTUNA_DB_PASSWORD'],
                             host=os.environ['OPTUNA_DB_HOST'],
                             port=os.environ['OPTUNA_DB_PORT'],
                             database=os.environ['OPTUNA_DB_NAME'])
    storage = optuna.storages.RDBStorage(url=str(storage_url))
    optuna.delete_study(study_name=study_name, storage=storage)
