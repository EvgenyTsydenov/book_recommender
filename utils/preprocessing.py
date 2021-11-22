from typing import Optional, Any

import pandas as pd
from isbnlib import to_isbn13
from langcodes import Language


def convert_to_isbn13(value: Any) -> Optional[str]:
    """Convert a value to ISBN-13.

    If it is impossible to convert the value to ISBN-13,
    None will be returned.

    :param: value: value to convert.
    :return: converted value.
    """
    new_value = to_isbn13(str(value))
    return new_value or None


def get_weighted_rating(df: pd.DataFrame,
                        min_rate_count: float,
                        mean_rate: float) -> float:
    """Calculate weighted rating.

    Use information about average rating and the number of votes
    it has accumulated.

    :param df: dataframe with ratings of an item.
    :param min_rate_count: minimum rating count required to be listed in the chart.
    :param mean_rate: mean rating across the whole ratings.
    :return: rating
    """
    rates = df['rating'].dropna()
    ratings_count = len(rates)
    ratings_mean = rates.mean()
    first = ratings_count * ratings_mean / (ratings_count + min_rate_count)
    second = min_rate_count * mean_rate / (ratings_count + min_rate_count)
    return first + second


def normalize_language_code(language_code: Any) -> Optional[str]:
    """Normalize language code.

    :param language_code: value to normalize.
    :return: normalized value or None.
    """
    try:
        return Language.get(language_code).language_name()
    except (ValueError, AttributeError):
        return None
