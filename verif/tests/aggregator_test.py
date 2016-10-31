import unittest
import verif.aggregator
import numpy as np


class TestAggregator(unittest.TestCase):
   def test_simple(self):
      agg = verif.aggregator.Mean()
      self.assertEqual(2, agg(np.array([1,2,3])))

if __name__ == '__main__':
   unittest.main()
