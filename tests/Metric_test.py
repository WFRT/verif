import unittest
import verif.Metric as Metric

class MyTest(unittest.TestCase):
   def test_func(self):
      ets = Metric.Ets()
      value = ets.calc(1,0,0,1)
      self.assertEqual(value, 1, "failed test")
   def test_func2(self):
      self.assertTrue(1, 2)


if __name__ == '__main__':
   unittest.main()
