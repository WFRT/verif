import inspect
import numpy as np
import sys

import verif.util


def get_all():
    """ Returns a list of all aggregator classes """
    temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return [i[1] for i in temp if i[0] != "Aggregator"]


def get(name):
    """ Returns an instance of an object with the given class name

    Arguments:
       name (str): The name of the class. Use a number between 0 and 1 to get the
          corresponding quantile aggregator
    """
    aggregators = get_all()
    a = None
    for aggregator in aggregators:
        if name == aggregator.name():
            a = aggregator()
    if a is None and verif.util.is_number(name):
        a = Quantile(float(name))

    if a is None:
        verif.util.error("No aggregator by the name '%s'" % name)

    return a


class Aggregator(object):
    """ Base class for aggregating an array (computing a scalar value from an array)

    Usage:
       mean = verif.aggregator.Mean()
       mean(np.array([1,2,3]))

    Attributes:
       name: A string representing the name of the aggregator
    """

    def __call__(self, array, axis=None):
        """ Compute the aggregated value. Returns a scalar value.

        Arguments:
           array (np.array): A 1D numpy array
           axis (int): Axis to aggregate across
        """
        raise NotImplementedError()

    @classmethod
    def name(cls):
        return cls.__name__.lower()

    def __hash__(self):
        # TODO
        return 1

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __ne__(self, other):
        return not self.__eq__(other)


class Mean(Aggregator):
    def __call__(self, array, axis=None):
        return np.mean(array, axis=axis)


class Median(Aggregator):
    def __call__(self, array, axis=None):
        return np.median(array, axis=axis)


class Min(Aggregator):
    def __call__(self, array, axis=None):
        return np.min(array, axis=axis)


class Max(Aggregator):
    def __call__(self, array, axis=None):
        return np.max(array, axis=axis)


class Std(Aggregator):
    def __call__(self, array, axis=None):
        return np.std(array, axis=axis)


class Variance(Aggregator):
    def __call__(self, array, axis=None):
        return np.var(array, axis=axis)


class Iqr(Aggregator):
    def __call__(self, array, axis=None):
        return np.percentile(array, 75, axis=axis) - np.percentile(array, 25, axis=axis)


class Range(Aggregator):
    def __call__(self, array, axis=None):
        return verif.util.nprange(array, axis=axis)


class Count(Aggregator):
    def __call__(self, array, axis=None):
        return verif.util.numvalid(array, axis=axis)


class Sum(Aggregator):
    def __call__(self, array, axis=None):
        return np.sum(array, axis=axis)


class Meanabs(Aggregator):
    """ The mean of the absolute values of the array """
    def __call__(self, array, axis=None):
        return np.mean(np.abs(array), axis=axis)


class Absmean(Aggregator):
    """ Absolute value of the mean of the array """
    def __call__(self, array, axis=None):
        return np.abs(np.mean(array, axis=axis))


class Quantile(Aggregator):
    def __init__(self, quantile):
        """ Returns a certain quantile from the array

        Arguments:
           quantile (float): A value between 0 and 1 inclusive
        """
        self.quantile = quantile
        if self.quantile < 0 or self.quantile > 1:
            verif.util.error("Quantile must be between 0 and 1")

    def __call__(self, array, axis=None):
        return np.percentile(array, self.quantile*100, axis=axis)

    def __eq__(self, other):
        return (self.__class__ == other.__class__) and (self.quantile == other.quantile)


class Change(Aggregator):
    """Difference between the last and the first element. Is most useful
    when used with -Tagg since it implies an array representing a sequence
    such as time.
    """
    def __call__(self, array, axis=None):
        if axis is None:
            return array.flatten()[-1] - array.flatten()[0]
        elif axis == 0:
            return array[-1, ...] - array[0, ...]
        elif axis == 1:
            return array[:, -1, ...] - array[:, 0, ...]
        elif axis == 2:
            return array[:, :, -1, ...] - array[:, :, 0, ...]
        elif axis == 3:
            return array[:, :, :, -1, ...] - array[:, :, :, 0, ...]
        elif axis == 4:
            return array[:, :, :, :, -1, ...] - array[:, :, :, :, 0, ...]
        else:
            raise NotImplementedError(f"This function not implemented for axis {axis}")
