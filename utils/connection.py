import functools
import sys
import time
from typing import Callable

from requests import RequestException


def retry_request(func: Callable) -> Callable:
    """Decorator for repeating requests.

    :param: func: function to decorate.
    :return: decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        for timeout in [1, 10, 30, 60, 90]:
            try:
                return func(*args, **kwargs)
            except RequestException as err:
                error = err
                exc_type, _, _ = sys.exc_info()
                print(f'Failed with {exc_type.__name__}.')
                time.sleep(timeout)
        print('The number of attempts is over.')
        raise error

    return _wrapper
