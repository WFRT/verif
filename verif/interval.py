import sys
import numpy as np


class Interval(object):
    """ Represents mathematical intervals in the real number line

    The following forms are supported, where [,] are inclusive and (,) are not:
    [lower, upper], (lower, upper], [lower, upper), or (lower, upper)

    Attributes:
       lower (float): The lower boundary of the interval
       upper (float): The upper boundary of the interval
       lower_eq (bool): True to include the lower boundary in the interval
       upper_eq (bool): True to include the upper boundary in the interval
    """
    def __init__(self, lower, upper, lower_eq, upper_eq):
        self.lower = lower if lower is not None else -np.inf
        self.upper = upper if upper is not None else np.inf
        self.lower_eq = lower_eq
        self.upper_eq = upper_eq

    def within(self, x):
        """ Is one or more values within the interval?

        Args:
           x (float or np.array): value(s)

        Returns:
           bool or np.array(bool): True if the value(s) is in the interval, False otherwise.
           If any of the inputs have np.nan values, then a masked array is
           returned with the corresponding answers masked out.
        """
        if isinstance(x, np.ndarray):
            I = np.where(np.isnan(x) == 0)
            """
            Only evaluate valid values, in order to avoid a warning. Then fill
            these values into a masked array
            """
            values = np.zeros(x.shape, bool)
            is_above = (x[I] > self.lower) | (self.lower_eq and x[I] == self.lower)
            is_below = (x[I] < self.upper) | (self.upper_eq and x[I] == self.upper)
            values[I] = is_above & is_below
            values = np.ma.masked_array(values, mask=np.isnan(x))
        else:
            if np.isnan(x):
                values = np.nan
            else:
                is_above = (x > self.lower) | (self.lower_eq and x == self.lower)
                is_below = (x < self.upper) | (self.upper_eq and x == self.upper)
                values = (is_above & is_below)

        return values

    @property
    def center(self):
        """ float: The center of the interval """
        if self.lower == -np.inf and self.upper == np.inf:
            return 0
        elif self.lower == -np.inf:
            return self.upper
        elif self.upper == np.inf:
            return self.lower
        else:
            return (self.lower + self.upper)/2

    def __str__(self):
        lower_bracket = "("
        if self.lower_eq:
            lower_bracket = "["
        upper_bracket = ")"
        if self.upper_eq:
            upper_bracket = "]"
        return "%s%f, %f%s" % (lower_bracket, self.lower, self.upper, upper_bracket)

    def __eq__(self, other):
        return self.lower == other.lower and\
               self.upper == other.upper and\
               self.lower_eq == other.lower_eq and\
               self.upper_eq == other.upper_eq

    def __ne__(self, other):
        return not self.__eq__(other)
