from typing import Any, Dict, Optional

from countryinfo import CountryInfo
from pycountry import countries


def parse_address_photon(place_info: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Photon location info.

    :param: place_info: address info.
    :return: country and coordinates.
    """
    if not place_info:
        return {}

    # Coordinates
    info = {
        'lon': (place_info['geometry']['coordinates'])[0],
        'lat': (place_info['geometry']['coordinates'])[1]
    }

    # Country name
    address = place_info['properties']
    country = address.get('country')
    if (country is None) and (address['osm_value'] in ['country']):
        country = address['name']
    info['country'] = country
    return info


def parse_address_positionstack(place_info: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Position Stack location info.

    :param: place_info: address info.
    :return: country and coordinates.
    """
    if not place_info:
        return {}

    info = {
        'lon': place_info.get('longitude'),
        'lat': place_info.get('latitude'),
        'country': place_info.get('country')
    }
    return info


def parse_address_opencage(place_info: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Open Cage location info.

    :param: place_info: address info.
    :return: country and coordinates.
    """
    if not place_info:
        return {}

    info = {
        'lon': place_info['geometry']['lng'],
        'lat': place_info['geometry']['lat']
    }
    country = place_info['components'].get('country')
    if not country:
        country = place_info['components'].get('ISO_3166-1_alpha-2')
    info['country'] = country
    return info


def parse_address(place_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract country name and coordinates from geocoding information.

    :param: place_info: address info.
    :return: country and coordinates.
    """
    if not place_info:
        return {}

    engine = place_info['engine']
    if engine == 'photon':
        return parse_address_photon(place_info['data'])
    if engine == 'position_stack':
        return parse_address_positionstack(place_info['data'])
    if engine == 'open_cage':
        return parse_address_opencage(place_info['data'])


def get_countrycode_countryinfo(name: str) -> Optional[str]:
    """Get country code by name with countryinfo package.

    :param: name: country name.
    :return: country code.
    """
    country = CountryInfo(name)
    code = None
    try:
        code = country.iso()['alpha2']
    finally:
        return code


def get_countrycode_pycountry(name: str) -> Optional[str]:
    """Get country code by name with pycountry package.

    :param: name: country name.
    :return: country code.
    """
    code = None
    try:
        code = countries.lookup(name).alpha_2
    finally:
        return code


def get_countrycode(name: str) -> Optional[str]:
    """Get country code by name.

    :param: name: country name.
    :return: Alpha-2 country code.
    """
    code = get_countrycode_countryinfo(name)
    if not code:
        code = get_countrycode_pycountry(name)
    return code
