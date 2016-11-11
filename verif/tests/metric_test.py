import unittest
import verif.metric
import verif.metric_type
import numpy as np


class MyTest(unittest.TestCase):
   def test_func(self):
      metric = verif.metric.Mae()
      metric.aggregator = verif.aggregator.Mean()
      value = metric.compute_from_obs_fcst(np.array([2, 1, 2]), np.array([2, 3, 1]))
      self.assertEqual(value, 1)
      value = metric.compute_from_obs_fcst(np.array([-2]), np.array([2]))
      self.assertEqual(value, 4)
      value = metric.compute_from_obs_fcst(np.array([]), np.array([]))
      self.assertTrue(np.isnan(value))
      value = metric.compute_from_obs_fcst(np.array([2, np.nan, 2]), np.array([2, 3, 1]))
      self.assertEqual(value, 0.5)

   def test_get_all_by_type(self):
      types = verif.metric.get_all_by_type(verif.metric_type.Deterministic())

   def test_name(self):
      mae = verif.metric.Mae()
      self.assertEqual("MAE", mae.name())

if __name__ == '__main__':
   unittest.main()
