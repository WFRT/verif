import unittest
import verif.metric
import numpy as np


class MyTest(unittest.TestCase):
   def test_func(self):
      obsSet = [[1, 2, 3],
                [1, 2],
                [1]]
      fcstSet = [[0, 2, 8],
                [3, 2],
                [2]]
      metrics = [verif.metric.Mae(), verif.metric.Bias(), verif.metric.Ef()]
      # Metrics in the inner lists, datasets in the outer
      expSet = [[2, -4.0 / 3, 1.0 / 3],
                [1, -1, 0.5],
                [1, -1, 1]]
      for s in range(0, len(obsSet)):
         obs = obsSet[s]
         fcst = fcstSet[s]
         for i in range(0, len(metrics)):
            metric = metrics[i]
            metric.aggregator = verif.aggregator.Mean()
            expected = expSet[s][i]
            value = metric.compute_from_obs_fcst(np.array(obs), np.array(fcst))
            message = metric.name() + " gives " + str(value) + " value for set " + str(s) + " (expected " + str(expected) + ")"
            self.assertAlmostEqual(value, expected, msg=message)

if __name__ == '__main__':
   unittest.main()
