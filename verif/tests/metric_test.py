import unittest
import verif.metric
import verif.metric_type
import numpy as np


def get():
   inputs = [verif.input.get_input(file) for file in ["verif/tests/files/raw.txt", "verif/tests/files/kf.txt"]]
   data = verif.data.Data(inputs=inputs)
   return data


class MetricTest(unittest.TestCase):
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

   def test_reample(self):
      metric = verif.metric.Ets()

      obs = np.random.randn(10000)
      fcst = np.random.randn(10000)
      othreshold = verif.interval.Interval(0.1, np.inf, True, True)
      fthresholds = [verif.interval.Interval(i, np.inf, True, True) for i in np.random.randn(1)]

      # Check that this runs
      q = np.zeros(len(fthresholds), float)
      for i in range(0, len(fthresholds)):
         q[i] = metric.compute_from_obs_fcst_resample(obs, fcst, 1, othreshold, fthresholds[i])
      for i in range(0, len(fthresholds)):
         q[i] = metric.compute_from_obs_fcst(obs, fcst, fthresholds[i])


class TestObsFcstBased(unittest.TestCase):
   def test_basic(self):
      data = get()
      # Try ObsFcstBased metrics here only
      metrics = [verif.metric.Rmse(),
                 verif.metric.Mae(),
                 verif.metric.Bias(),
                 verif.metric.Corr(),
                 verif.metric.Dmb(),
                 verif.metric.Mbias(),
                 verif.metric.Ef(),
                 ]
      obs = [[0, 1.5, 2], [0]]  # Case1, Case 2, ...
      fcst = [[3.1, 1.1, -2.1], [1]]
      expected = [[2.976575213, 1],  # Rmse
                  [2.5333333333, 1],  # Mae
                  [-0.466666666666, 1],  # Bias
                  [-0.915724295, np.nan],  # Corr
                  [1.0/0.6, 0],  # Dmb
                  [0.6, np.nan],  # Mbias
                  [1.0/3, 1],  # Ef
                  ]

      for m in range(len(metrics)):
         metric = metrics[m]
         for i in range(0, len(obs)):
            value = metric.compute_from_obs_fcst(np.array(obs[i]), np.array(fcst[i]))
            if np.isnan(value):
               self.assertTrue(np.isnan(expected[m][i]))
            else:
               self.assertAlmostEqual(expected[m][i], value)


class TestThresholdBased(unittest.TestCase):
   def test_basic(self):
      """ Test the basic threshold-based metrics """
      data = get()
      metrics = [verif.metric.Within(),
                 verif.metric.A(),  # Hit
                 verif.metric.B(),  # FA
                 verif.metric.C(),  # Miss
                 verif.metric.D(),  # Correct rejection
                 verif.metric.Hit(),
                 verif.metric.Threat(),
                 verif.metric.Conditional(),
                 verif.metric.XConditional(func=np.median),
                 ]
      intervals = [verif.interval.Interval(-np.inf, 0, True, True),  # [-inf, 0]
                   verif.interval.Interval(-np.inf, 1, True, True),
                   verif.interval.Interval(-np.inf, 2, True, True),
                   ]
      obs = [0, 1.5, 2]
      fcst = [3.1, 1.1, -2.1]
      N = len(obs)*1.0

      # Each line is one metric (one number for each threshold)
      expected = [[0/N, 100/N, 100/N],  # Within
                  [0/N, 0/N, 2/N],  # Hit
                  [1/N, 1/N, 0/N],  # FA
                  [1/N, 1/N, 1/N],  # Miss
                  [1/N, 1/N, 0/N],  # Correct rejection
                  [0, 0, 2.0/3],  # Hit rate
                  [0, 0, 2.0/3],  # Threat score
                  [3.1, 3.1, 0.7],  # Average fcst given obs in interval
                  [0, 0, 1.5],  # Average obs given obs in interval
                  ]

      for m in range(len(metrics)):
         metric = metrics[m]
         for i in range(len(intervals)):
            value = metric.compute_from_obs_fcst(np.array(obs), np.array(fcst), intervals[i])
            ex = expected[m][i] * 1.0
            if np.isnan(value):
               self.assertTrue(np.isnan(ex))
            else:
               self.assertAlmostEqual(ex, value)

   def test_threat(self):
      """ An extra test for Threat score """
      metric = verif.metric.Threat()
      obs = np.array([0, 1, 2, 3])
      fcst = np.array([0, 3, 1, 2])

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
