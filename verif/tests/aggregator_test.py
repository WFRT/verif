import unittest
import verif.aggregator
import numpy as np


class TestAggregator(unittest.TestCase):
   def test_simple(self):
      agg = verif.aggregator.Mean()
      self.assertEqual(2, agg(np.array([1,2,3])))

   def test_name(self):
      mean = verif.aggregator.Mean()
      self.assertEqual("mean", mean.name())
      mean2 = verif.aggregator.get("mean")
      self.assertEqual("mean", mean2.name())
      self.assertEqual(mean, mean2)

if __name__ == '__main__':
   unittest.main()
