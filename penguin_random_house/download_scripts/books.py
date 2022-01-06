import json
import multiprocessing
import os
import tempfile
from typing import Union

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests import RequestException
from tqdm import tqdm

from utils.connection import retry_request

# Load environment variables
load_dotenv()

# Specify common values
DOMAIN = os.environ['PENGUIN_DOMAIN']
POINT = 'https://api.penguinrandomhouse.com'
REQUEST_HEADERS = {'Accept': 'application/json'}
REQUEST_PARAMS = {
    'api_key': os.environ['PENGUIN_API_KEY'],
    'suppressLinks': True,
    'returnEmptyLists': True,
    'suppressRecordCount': True
}


@retry_request
def get_book_data(isbn: Union[int, str]) -> dict:
    """Get book data by ISBN-13.

    :param: index: book isbn.
    :return: book data.
    """
    # Create the URL
    # Include the information about authors, series, category in the response
    url = f'{POINT}/resources/v2/title/domains/{DOMAIN}/titles/{isbn}?' \
          f'zoom={POINT}/title/authors/definition' \
          f'&zoom={POINT}/title/categories/definition' \
          f'&zoom={POINT}/title/titles/content/definition' \
          f'&zoom={POINT}/title/series/definition' \
          f'&zoom={POINT}/title/works/definition'

    # Request
    response = requests.get(url, params=REQUEST_PARAMS,
                            headers=REQUEST_HEADERS)

    # If error
    if not response:
        # Raise exception to retry request by decorator
        raise RequestException()
    return response.json()


def worker(path_isbns: str, path_data: str, worker_id: int = 0) -> None:
    """Download book info in workers.

    :param: path_data: path where to save book info.
    :param: path_isbns: path where to get ISBNs.
    :param: worker_id: worker id if executed in several processes.
    """
    # Show progress bar only for the first worker
    tqdm_disable = worker_id != 0

    # ISBNs to download
    isbns = np.load(path_isbns)

    # File where to save the result
    with open(path_data, 'w') as file:

        # Start downloading
        for isbn in tqdm(isbns, desc=f'worker_{worker_id}',
                         position=worker_id, disable=tqdm_disable):

            # Download
            books_info = get_book_data(isbn)

            # If request failed
            if books_info.get('status') != 'ok':
                # Print error and params
                print(books_info.get('error'))
                print(books_info.get('params'))
                continue

            # If request was successful, save data
            data = books_info.get('data')
            if data:
                data_string = json.dumps(data)
                file.write(data_string)
                file.write('\n')


if __name__ == '__main__':

    # ISBN to download
    path_ratings = os.path.join('..', 'data_interm', 'ratings_joined.csv')
    ratings = pd.read_csv(path_ratings, usecols=['isbn13'])
    isbns_unique = ratings['isbn13'].unique()

    # Start downloading
    # Since the downloading is slow, split this into several processes
    workers_count = 50
    with tempfile.TemporaryDirectory() as tmpdir:

        # Run processes
        processes = []
        for worker_id, worker_isbns in enumerate(
                np.array_split(isbns_unique, workers_count)):
            # Save ISBNs for worker
            worker_path_isbns = os.path.join(tmpdir, f'isbns_{worker_id}.npy')
            np.save(worker_path_isbns, worker_isbns)

            # Run process
            worker_path_data = os.path.join(tmpdir, f'data_{worker_id}.txt')
            process = multiprocessing.Process(
                target=worker, args=(worker_path_isbns, worker_path_data,
                                     worker_id))
            processes.append(process)
            process.start()
        for proc in processes:
            proc.join()

        # Concat result of workers
        path_books_raw = os.path.join('..', 'data_raw', 'books.txt')
        with open(path_books_raw, 'w') as outfile:
            for chunk in os.listdir(tmpdir):
                if not chunk.startswith('data'):
                    continue
                path_chunk = os.path.join(tmpdir, chunk)
                with open(path_chunk, 'r') as infile:
                    outfile.write(infile.read())
