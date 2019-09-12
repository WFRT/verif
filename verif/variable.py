import re
import matplotlib.ticker


def guess_x0(name):
    """
    Attempt to automatically detect the value of the lower discrete mass
    (e.g. 0 mm for precipitation)
    """
    prog = re.compile(".*precip.*")
    if prog.match(name.lower()):
        return 0
    prog = re.compile("RH")
    if prog.match(name):
        return 0
    return None


def guess_x1(name):
    """
    Attempt to automatically detect the value of the upper discrete mass
    (e.g. 100 % for RH)
    """
    prog = re.compile("RH")
    if prog.match(name):
        return 100
    return None


class Variable(object):
    """ Represents information about the forecast variable (e.g. temperature)

    Attributes:
       name (str): Name of the variable
       units (str): Units of the variable
       x0 (float): The value for the lower discrete mass (e.g. 0 mm for precipitation)
       x1 (float): The value for the upper discrete mass (e.g. 100 % for RH)
    """
    def __init__(self, name, units, formatter=matplotlib.ticker.ScalarFormatter(), x0=None, x1=None):
        self.name = name
        self.units = units
        self.formatter = formatter
        self.x0 = x0
        self.x1 = x1
