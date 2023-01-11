"""
Miscellaneous objects.
"""

from typing import Iterable


class Singleton(int):
    """
    Abstract base class for classes whose objects are singletons. Only
    works for values between -5 and 256, which are initialised as
    references to an existing object in python.
    """

    def __new__(cls, value, name=None, short_name=None):
        if name is None:
            name = str(value)

        s = int.__new__(cls, value)
        s.value = value
        s.name = name
        s.short_name = short_name
        
        return s

    def __repr__(self):
        return self.name


def _flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def flatten(xs):
    return type(xs)(_flatten(xs))
