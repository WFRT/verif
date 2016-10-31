import matplotlib.ticker


class Variable(object):
   """ Represents information about the forecast variable (e.g. temperature)

   Attributes:
   name:       Name of the variable
   units:      Units of the variable
   x0:         The value for the lower discrete mass (e.g. 0 mm for precipitation)
   x1:         The value for the upper discrete mass (e.g. 100 % for RH)
   """
   def __init__(self, name, units, formatter=matplotlib.ticker.ScalarFormatter(), x0=None, x1=None):
      self.name = name
      self.units = units
      self.formatter = formatter
      self.x0 = x0
      self.x1 = x1
