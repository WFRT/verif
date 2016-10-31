import re
import matplotlib.ticker


class Variable(object):
   """ Represents information about the forecast variable (e.g. temperature)

   Attributes:
   name:       Name of the variable
   units:      Units of the variable
   """
   def __init__(self, name, units, formatter=matplotlib.ticker.ScalarFormatter()):
      self.name = name
      self.units = units
      self.formatter = formatter

   def get_x0(self):
      """ Get the value for the lower discrete mass (e.g. 0 mm for precipitation) """
      x0 = None
      prog = re.compile("Precip.*")
      if(prog.match(self.name)):
         x0 = 0
      return x0

   def get_x1(self):
      """ Get the value for the upper discrete mass (e.g. 100 % for RH) """
      x1 = None
      prog = re.compile("RH")
      if(prog.match(self.name)):
         x1 = 100
      return x1
