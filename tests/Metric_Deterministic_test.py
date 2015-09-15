import unittest
import verif.Metric as Metric
import numpy as np


class MyTest(unittest.TestCase):
   def test_func(self):
      obsSet = [[1, 2, 3],
                [1, 2],
                [1]]
      fcstSet = [[0, 2, 8],
                [3, 2],
                [2]]
      metrics = [Metric.Mae(), Metric.Bias(), Metric.Ef()]
      # Metrics in the inner lists, datasets in the outer
      expSet = [[2, -4.0 / 3, 100.0 / 3],
                [1, -1, 50],
                [1, -1, 100]]
      for s in range(0, len(obsSet)):
         obs = obsSet[s]
         fcst = fcstSet[s]
         for i in range(0, len(metrics)):
            metric = metrics[i]
            metric.setAggregator("mean")
            expected = expSet[s][i]
            value = metric.computeObsFcst(np.array(obs), np.array(fcst))
            message = metric.name() + " gives " + str(value) + " value for set " + str(s) + " (expected " + str(expected) + ")"
            self.assertAlmostEqual(value, expected, msg=message)

if __name__ == '__main__':
   unittest.main()
