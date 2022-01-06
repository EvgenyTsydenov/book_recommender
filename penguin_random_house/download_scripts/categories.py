import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv
from requests import RequestException

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


def get_category_info(category_id: str) -> dict:
    """Get information about category.

    :param: category_id: category id.
    :return: information.
    """
    # Create the URL
    url = f'{POINT}/resources/v2/title/domains/{DOMAIN}/' \
          f'categories/{category_id}'

    # Request
    response = requests.get(url, params=REQUEST_PARAMS,
                            headers=REQUEST_HEADERS)

    # If error
    if not response:
        # Raise exception to retry request by decorator
        raise RequestException()

    # Extract data
    category_data = response.json().get('data')
    if category_data:
        return category_data['categories'][0]
    return {}


def get_children_info(category_id: str) -> list[dict]:
    """Get information about children categories of the current category.

    :param: category_id: category id.
    :return: info about children categories.
    """
    # Create the URL
    url = f'{POINT}/resources/v2/title/domains/{DOMAIN}/' \
          f'categories/{category_id}/children'

    # Request
    response = requests.get(url, params=REQUEST_PARAMS,
                            headers=REQUEST_HEADERS)

    # If error
    if not response:
        # Raise exception to retry request by decorator
        raise RequestException()

    # Extract data
    children_data = response.json().get('data')
    if children_data:
        return children_data['categories']
    return []


def get_all_categories(category_id: str, data: Optional[list] = None) -> dict:
    """Get information about all categories and save it to json.

    :param: category_id: category id.
    :param: path_save: where to save result.
    """
    if data is None:
        data = []

    for child_info in get_children_info(category_id):
        # Save
        child_info['parent'] = category_id
        data.append(child_info)

        # Find children
        data = get_all_categories(child_info['catId'], data)

    return data


if __name__ == '__main__':
    # Where to save
    path_cats_raw = os.path.join('..', 'data_raw', 'categories.txt')

    # The parent category
    top_category_id = '2000000000'
    top_category_info = get_category_info(top_category_id)
    data = [top_category_info]

    # Load information about all categories
    data.extend(get_all_categories(top_category_id))

    # Save
    os.makedirs(os.path.dirname(path_cats_raw), exist_ok=True)
    with open(path_cats_raw, 'w') as file:
        for row in data:
            data_string = json.dumps(row)
            file.write(data_string)
            file.write('\n')
