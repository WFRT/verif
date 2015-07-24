import unittest

class MyTest(unittest.TestCase):
   def test_func(self):
      self.assertTrue(0, "failed test")
   def test_func2(self):
      self.assertTrue(1, 2)


if __name__ == '__main__':
   unittest.main()
