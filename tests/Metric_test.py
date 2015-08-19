import unittest
import verif.Metric as Metric
import numpy as np

class MyTest(unittest.TestCase):
   def test_func(self):
      metric = Metric.Mae()
      metric.setAggregator("mean")
      value = metric.computeObsFcst(np.array([2,1,2]),np.array([2,3,1]))
      self.assertEqual(value, 1)
      value = metric.computeObsFcst(np.array([-2]),np.array([2]))
      self.assertEqual(value, 4)
      value = metric.computeObsFcst(np.array([]),np.array([]))
      self.assertTrue(np.isnan(value))
      value = metric.computeObsFcst(np.array([2,np.nan,2]),np.array([2,3,1]))
      self.assertEqual(value, 0.5)

if __name__ == '__main__':
   unittest.main()
