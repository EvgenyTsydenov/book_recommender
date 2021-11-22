import functools
import time
from typing import Callable

from requests import RequestException


def retry_request(func: Callable) -> Callable:
    """"Decorator for repeating requests.

    :param: func: function to decorate.
    :return: decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        for timeout in [1, 10, 30, 60, 90]:
            try:
                return func(*args, **kwargs)
            except RequestException:
                time.sleep(timeout)
                continue
        return {}

    return _wrapper
