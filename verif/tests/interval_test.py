import unittest
import verif.interval


class MyTest(unittest.TestCase):
   _tests = [-0.1,0,0.1,0.9,1,1.1]
   def compare(self, interval, expected):
      """ Run interval.within on all elements in _tests and compare the result with expected """
      for i in range(0, len(self._tests)):
         self.assertEqual(expected[i], interval.within(self._tests[i]))

   def test_1(self):
      interval = verif.interval.Interval(0, 1, False, False)  # 0 < x < 1
      self.compare(interval, [False, False, True, True, False, False])

   def test_2(self):
      interval = verif.interval.Interval(0, 1, True, True)  # 0 <= x <= 1
      self.compare(interval, [False, True, True, True, True, False])

   def test_3(self):
      interval = verif.interval.Interval(0, 1, False, True)  # 0 < x <= 1
      self.compare(interval, [False, False, True, True, True, False])

   def test_4(self):
      interval = verif.interval.Interval(0, 1, True, False)  # 0 <= x < 1
      self.compare(interval, [False, True, True, True, False, False])

   def test_zero_interval_1(self):
      interval = verif.interval.Interval(0, 0, False, False)  # 0 < x < 0
      self.compare(interval, [False, False, False, False, False, False])

   def test_zero_interval_2(self):
      interval = verif.interval.Interval(0, 0, True, True)  # 0 <= x <= 0
      self.compare(interval, [False, True, False, False, False, False])

   def test_zero_interval_3(self):
      interval = verif.interval.Interval(0, 0, False, True)  # 0 < x <= 0
      self.compare(interval, [False, False, False, False, False, False])

   def test_zero_interval_4(self):
      interval = verif.interval.Interval(0, 0, True, False)  # 0 <= x < 0
      self.compare(interval, [False, False, False, False, False, False])

   def test_equal(self):
      from verif.interval import Interval
      self.assertEqual(Interval(0,0,True,False), Interval(0,0,True,False))
      self.assertTrue(Interval(0,0,True,False) == Interval(0,0,True,False))
      self.assertFalse(Interval(0,0,True,False) != Interval(0,0,True,False))

   def test_unequal(self):
      from verif.interval import Interval
      for interval in [Interval(0,0,True,True), Interval(0,0,False,False),
            Interval(0,1,True,False), Interval(1,0,True,False)]:
         self.assertFalse(Interval(0,0,True,False) == interval)
         self.assertFalse(Interval(0,0,True,False) == interval)
         self.assertTrue(Interval(0,0,True,False) != interval)
         self.assertTrue(Interval(0,0,True,False) != interval)


if __name__ == '__main__':
   unittest.main()
