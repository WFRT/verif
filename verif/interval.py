import sys
import numpy as np
import verif.driver


class Interval(object):
   """ Represents an intervals in the form:
   [lower, upper]
   (lower, upper]
   [lower, upper)
   (lower, upper)
   
   Attributes:
   lower             The lower boundary of the interval
   upper             The upper boundary of the interval
   lower_equality    Should the lower boundary be included in the interval?
   upper_equality    Should the upper boundary be included in the interval?
   """
   def __init__(self, lower, upper, lower_equality, upper_equality):
      self.lower = lower if lower is not None else -np.inf
      self.upper = upper if upper is not None else np.inf
      self.lower_equality = lower_equality
      self.upper_equality = upper_equality

   def within(self, x):
      is_above = (x > self.lower) | (self.lower_equality and x == self.lower)
      is_below = (x < self.upper) | (self.upper_equality and x == self.upper)

      return is_above & is_below

   @property
   def center(self):
      if self.lower == -np.inf and self.upper == np.inf:
         return None
      elif self.lower == -np.inf:
         return self.upper
      elif self.upper == np.inf:
         return self.lower
      else:
         return (self.lower + self.upper)/2

   def __str__(self):
      lower_bracket = "("
      if self.lower_equality:
         lower_bracket = "["
      upper_bracket = ")"
      if self.upper_equality:
         upper_bracket = "]"
      return "%s%f, %f%s" % (lower_bracket, self.lower, self.upper, upper_bracket)

   def __eq__(self, other):
      return self.lower == other.lower and\
             self.upper == other.upper and\
             self.lower_equality == other.lower_equality and\
             self.upper_equality == other.upper_equality
