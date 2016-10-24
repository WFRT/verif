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
class Threshold(Field):
   def __init__(self, threshold):
      self.threshold = threshold
class Spread(Field):
   pass
