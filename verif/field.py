class Field(object):
   @classmethod
   def name(cls):
      name = cls.__name__
      return name


class Obs(Field):
   pass


class Deterministic(Field):
   pass


class Ensemble(Field):
   def __init__(self, member):
      self.member = member


class Quantile(Field):
   def __init__(self, quantile):
      self.quantile = quantile

   def name(self):
      return "Quantile(%g)" % self.quantile

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self.quantile == other.quantile


class Threshold(Field):
   def __init__(self, threshold):
      self.threshold = threshold

   def name(self):
      return "Threshold(%g)" % self.threshold

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self.threshold == other.threshold


class Spread(Field):
   pass


class Other(Field):
   def __init__(self, name):
      self._name = name

   def name(self):
      return "Other(%s)" % self._name

   def __eq__(self, other):
      if self.__class__ != other.__class__:
         return False
      return self._name == other._name
