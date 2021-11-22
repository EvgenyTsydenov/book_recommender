import os
import pickle
from typing import Any, Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from utils.connection import retry_request

# Load environment variables
load_dotenv()


@retry_request
def find_address_positionstack(place: str, limit: int = 1,
                               timeout: int = 30) -> Dict[str, Any]:
    """Get location info with Position Stack geocoder.

    Limit is 25000 requests per month without a fee.

    :param: place: address.
    :param: limit: number of results to return.
    :param: timeout: connection timeout.
    :return: parsed address.
    """
    params = {
        'access_key': os.environ['POSITIONSTACK_API_KEY'],
        'query': place,
        'limit': limit
    }
    r = requests.get(url='http://api.positionstack.com/v1/forward?/',
                     params=params, timeout=timeout)
    res = r.json().get('data')
    if res:
        return res[0]
    return {}


@retry_request
def find_address_photon(place: str, lang: str = 'en',
                        limit: int = 1, timeout: int = 30) -> Dict[str, Any]:
    """Get location info with Photon geocoder.

    There is no limit.

    :param: place: address.
    :param: lang: language.
    :param: limit: number of results to return.
    :param: timeout: connection timeout.
    :return: parsed address.
    """
    params = {
        'q': place,
        'lang': lang,
        'limit': limit
    }
    r = requests.get(url='https://photon.komoot.io/api/',
                     params=params, timeout=timeout)
    res = r.json().get('features')
    if res:
        return res[0]
    return {}


@retry_request
def find_address_opencage(place: str, limit: int = 1,
                          timeout: int = 30) -> Dict[str, Any]:
    """Get location info with Open Cage geocoder.

    Limit is 2500 requests per day without a fee.

    :param: place: address.
    :param: limit: number of results to return.
    :param: timeout: connection timeout.
    :return: parsed address.
    """
    params = {
        'q': place,
        'key': os.environ['OPENCAGE_API_KEY'],
        'limit': limit, 'no_annotations': 1
    }
    r = requests.get(url='https://api.opencagedata.com/geocode/v1/json?',
                     params=params, timeout=timeout)
    res = r.json().get('results')
    if res:
        return res[0]
    return {}


def find_address(location: str) -> Dict[str, Dict]:
    """Find information about the location.

    :param location: location to parse.
    :return: information about location.
    """

    # First, geocode with Photon because it does not have limits
    data = find_address_photon(location)
    engine = 'photon'

    # Try to geocode with Position Stack
    if not data:
        data = find_address_positionstack(location)
        engine = 'position_stack'

    # Try to geocode with Open Cage
    if not data:
        data = find_address_opencage(location)
        engine = 'open_cage'

    # Save
    if data:
        return {'engine': engine, 'data': data}
    return {}


if __name__ == '__main__':

    # Import locations
    path = os.path.join('..', 'data_interm', 'unique_locations.pkl')
    unique_locations = pd.read_pickle(path)

    # Find locations
    geocoded = {}
    for location in tqdm(unique_locations):
        geocoded[location] = find_address(location)

    # Save the result
    path_geocoded = os.path.join('..', 'data_interm', 'geocoded_locations.pkl')
    with open(path_geocoded, 'wb') as file:
        pickle.dump(geocoded, file)
