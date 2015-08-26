import unittest
import verif.Common as Common
import numpy as np

class TestParseNumbers(unittest.TestCase):
   def test_simple(self):
      self.assertEqual([2], Common.parseNumbers("2"))
      with self.assertRaises(SystemExit):
         Common.parseNumbers("test")
   def test_vector(self):
      self.assertEqual([2,3,4,5], Common.parseNumbers("2:5"))
      self.assertEqual([], Common.parseNumbers("2:1"))
      with self.assertRaises(SystemExit):
         self.assertEqual([2], Common.parseNumbers("2:test"))
   def test_vectorInc(self):
      self.assertEqual([2,5,8], Common.parseNumbers("2:3:8"))
      self.assertEqual([2,5], Common.parseNumbers("2:3:7"))
      self.assertEqual([], Common.parseNumbers("2:-1:7"))
      self.assertEqual([2,1,0], Common.parseNumbers("2:-1:0"))
      self.assertEqual([8,5,2], Common.parseNumbers("8:-3:0"))
      with self.assertRaises(SystemExit):
         Common.parseNumbers("2:3:test")
         Common.parseNumbers("test:3:5")
         Common.parseNumbers("2:test:5")
   def test_comma(self):
      self.assertEqual([2,5], Common.parseNumbers("2,5"))
      self.assertEqual([3,3], Common.parseNumbers("3,3"))
      with self.assertRaises(SystemExit):
         Common.parseNumbers("test")
   def test_mix(self):
      self.assertEqual([3,4,3,2,3,4,5], Common.parseNumbers("3:4,3,2:5"))
      with self.assertRaises(SystemExit):
         Common.parseNumbers("2,5:8,3,test")
         Common.parseNumbers("2,5:8,test,5")
         Common.parseNumbers("2,5:test,3,5")
         Common.parseNumbers("2,5:test,3,5")
         Common.parseNumbers("2,test:8,3,5")
         Common.parseNumbers("test,5:8,3,5")
   def test_date(self):
      self.assertEqual([20141230,20141231,20150101,20150102,20150103], Common.parseNumbers("20141230:20150103", True))
      self.assertEqual([20141230,20150101,20150103], Common.parseNumbers("20141230:2:20150104", True))

class TestGetDate(unittest.TestCase):
   def test_simple(self):
      self.assertEqual(20150207, Common.getDate(20150206, 1))
      self.assertEqual(20150205, Common.getDate(20150206, -1))
      self.assertEqual(20150215, Common.getDate(20150210, 5))
   def test_endofyear(self):
      self.assertEqual(20150101, Common.getDate(20141231, 1))
      self.assertEqual(20141226, Common.getDate(20150105, -10))
      self.assertEqual(20150105, Common.getDate(20141226, 10))
      self.assertEqual(20141231, Common.getDate(20150101, -1))

if __name__ == '__main__':
   unittest.main()
