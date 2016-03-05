import re


# Represents information about the forecast variable
# (e.g. temperature)
class Variable(object):
   def __init__(self, name, units):
      self._name = name
      self._units = units

   def name(self, value=None):
      if value is None:
         return self._name
      else:
         self.name = value

   def units(self, value=None):
      if value is None:
         return self._units
      else:
         self._units = value

   # Get the value for the lower discrete mass
   # (e.g. 0 mm for precipitation)
   def getX0(self):
      x0 = None
      prog = re.compile("Precip.*")
      if(prog.match(self.name())):
         x0 = 0
      return x0

   # Get the value for the upper discrete mass
   # (e.g. 100 % for RH)
   def getX1(self):
      x1 = None
      prog = re.compile("RH")
      if(prog.match(self.name())):
         x1 = 100
      return x1
