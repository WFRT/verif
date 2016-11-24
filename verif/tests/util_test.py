import unittest
import verif.util
import verif
import numpy as np


class TestParseNumbers(unittest.TestCase):
   def test_simple(self):
      self.assertEqual([2], verif.util.parse_numbers("2"))
      with self.assertRaises(SystemExit):
         verif.util.parse_numbers("test")

   def test_vector(self):
      self.assertEqual([2, 3, 4, 5], verif.util.parse_numbers("2:5"))
      self.assertEqual([], verif.util.parse_numbers("2:1"))
      with self.assertRaises(SystemExit):
         self.assertEqual([2], verif.util.parse_numbers("2:test"))

   def test_vectorInc(self):
      self.assertEqual([2, 5, 8], verif.util.parse_numbers("2:3:8"))
      self.assertEqual([2, 5], verif.util.parse_numbers("2:3:7"))
      self.assertEqual([], verif.util.parse_numbers("2:-1:7"))
      self.assertEqual([2, 1, 0], verif.util.parse_numbers("2:-1:0"))
      self.assertEqual([8, 5, 2], verif.util.parse_numbers("8:-3:0"))
      with self.assertRaises(SystemExit):
         verif.util.parse_numbers("2:3:test")
         verif.util.parse_numbers("test:3:5")
         verif.util.parse_numbers("2:test:5")

   def test_comma(self):
      self.assertEqual([2, 5], verif.util.parse_numbers("2,5"))
      self.assertEqual([3, 3], verif.util.parse_numbers("3,3"))
      with self.assertRaises(SystemExit):
         verif.util.parse_numbers("test")

   def test_mix(self):
      self.assertEqual([3, 4, 3, 2, 3, 4, 5], verif.util.parse_numbers("3:4,3,2:5"))
      with self.assertRaises(SystemExit):
         verif.util.parse_numbers("2,5:8,3,test")
         verif.util.parse_numbers("2,5:8,test,5")
         verif.util.parse_numbers("2,5:test,3,5")
         verif.util.parse_numbers("2,5:test,3,5")
         verif.util.parse_numbers("2,test:8,3,5")
         verif.util.parse_numbers("test,5:8,3,5")

   def test_date(self):
      self.assertEqual([20141230, 20141231, 20150101, 20150102, 20150103], verif.util.parse_numbers("20141230:20150103", True))
      self.assertEqual([20141230, 20150101, 20150103], verif.util.parse_numbers("20141230:2:20150104", True))

   def test_convert_times(self):
      times = np.array([1331856000, -2180131200])
      times2 = verif.util.convert_times(times)
      from matplotlib.dates import *
      import datetime
      self.assertEqual(date2num(datetime.datetime(2012, 3, 16, 0, 0)), times2[0])
      self.assertEqual(date2num(datetime.datetime(1900, 12, 1, 0, 0)), times2[1])

   def test_date_to_unixtime(self):
      import time
      s = time.time()
      self.assertEqual(1475280000, verif.util.date_to_unixtime_slow(20161001))
      e = time.time()
      print (e - s)
      s = time.time()
      self.assertEqual(1475280000, verif.util.date_to_unixtime(20161001))
      e = time.time()
      print (e - s)

   def test_unixtime_to_date(self):
      self.assertEqual(20161001, verif.util.unixtime_to_date(1475280000))

   def test_apply_threshold(self):
      ar = np.array([1,3,2])
      np.testing.assert_array_equal(np.array([0,1,0]), verif.util.apply_threshold(ar, "above", 2))
      np.testing.assert_array_equal(np.array([0,1,1]), verif.util.apply_threshold(ar, "above=", 2))
      np.testing.assert_array_equal(np.array([1,0,0]), verif.util.apply_threshold(ar, "below", 2))
      np.testing.assert_array_equal(np.array([1,0,1]), verif.util.apply_threshold(ar, "below=", 2))

class TestGetDate(unittest.TestCase):
   def test_simple(self):
      self.assertEqual(20150207, verif.util.get_date(20150206, 1))
      self.assertEqual(20150205, verif.util.get_date(20150206, -1))
      self.assertEqual(20150215, verif.util.get_date(20150210, 5))

   def test_endofyear(self):
      self.assertEqual(20150101, verif.util.get_date(20141231, 1))
      self.assertEqual(20141226, verif.util.get_date(20150105, -10))
      self.assertEqual(20150105, verif.util.get_date(20141226, 10))
      self.assertEqual(20141231, verif.util.get_date(20150101, -1))


class TestGetIntervals(unittest.TestCase):
   def test_get_intervals(self):
      thresholds = [0,1,5]
      intervals = verif.util.get_intervals("above", thresholds)
      self.assertEqual(verif.interval.Interval(0,np.inf,False,False), intervals[0])

if __name__ == '__main__':
   unittest.main()
