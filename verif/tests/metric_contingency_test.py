import unittest
import verif.metric
import verif.metric_type
import verif.interval
import numpy as np


class ContingencyTest(unittest.TestCase):
   def test_func(self):
      metric = verif.metric.Ets()

      obs = np.random.randn(10000)
      fcst = np.random.randn(10000)
      othreshold = verif.interval.Interval(0.1,np.inf, True, True)
      fthresholds = [verif.interval.Interval(i, np.inf, True, True) for i in np.random.randn(1)]

      # Check that this runs
      q = np.zeros(len(fthresholds), float)
      for i in range(0, len(fthresholds)):
         q[i] = metric.compute_from_obs_fcst_resample(obs, fcst, 1, othreshold, fthresholds[i])
      for i in range(0, len(fthresholds)):
         q[i] = metric.compute_from_obs_fcst(obs, fcst, fthresholds[i])


class ThreatTest(unittest.TestCase):
   def test_func(self):
      metric = verif.metric.Threat()
      obs = np.array([0,1,2,3])
      fcst = np.array([0,3,1,2])

      # Hits: 1
      # FA: 1
      # Miss: 1
      # CR: 0
      interval = verif.interval.Interval(1.5, np.inf, True, True)
      f_interval = verif.interval.Interval(1.5, np.inf, True, True)
      value = metric.compute_from_obs_fcst(obs, fcst, interval, f_interval)
      self.assertEqual(value, 1.0/3)


if __name__ == '__main__':
   unittest.main()
