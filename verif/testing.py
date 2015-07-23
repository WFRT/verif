import unittest

class MyTest(unittest.TestCase):
   def test(self):
      self.assertTrue(1, 2)

def main():
   unittest.main()

if __name__ == '__main__':
  main()
