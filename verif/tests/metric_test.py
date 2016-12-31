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

class TestRmse(unittest.TestCase):
   def _get(self):
      inputs = [verif.input.get_input(file) for file in ["examples/raw.txt", "examples/kf.txt"]]
      data = verif.data.Data(inputs=inputs)
      return data

   def test_basic(self):
      data = self._get()
      metrics = [verif.metric.Rmse(),
                 verif.metric.Mae(),
                 verif.metric.Bias(),
                 verif.metric.Corr(),
                 verif.metric.Mbias(),
                 ]
      obs = [[0,1.5,2],[0]] # Case1, Case 2, ...
      fcst = [[3.1,1.1,-2.1],[1]]
      expected = [[2.976575213,1], # Rmse
                  [2.5333333333,1], # Mae
                  [-0.466666666666,1], # Bias
                  [-0.915724295,np.nan], # Corr
                  [0.6,np.nan], # Mbias
                  ]

      for m in range(0,len(metrics)):
         metric = metrics[m]
         for i in range(0, len(obs)):
            value = metric.compute_from_obs_fcst(np.array(obs[i]), np.array(fcst[i]))
            if np.isnan(value):
               self.assertTrue(np.isnan(expected[m][i]))
            else:
               self.assertAlmostEqual(expected[m][i], value)


if __name__ == '__main__':
   unittest.main()
