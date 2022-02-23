import os

import optuna
from dotenv import load_dotenv
from sqlalchemy.engine import URL

if __name__ == '__main__':
    load_dotenv()
    study_name = 'gradient_descent'
    storage_url = URL.create(drivername=os.environ['OPTUNA_DB_DRIVER'],
                             username=os.environ['OPTUNA_DB_USER'],
                             password=os.environ['OPTUNA_DB_PASSWORD'],
                             host=os.environ['OPTUNA_DB_HOST'],
                             port=os.environ['OPTUNA_DB_PORT'],
                             database=os.environ['OPTUNA_DB_NAME'])
    storage = optuna.storages.RDBStorage(url=str(storage_url))
    optuna.delete_study(study_name=study_name, storage=storage)
