import unittest
import verif.axis
import numpy as np


class TestAxis(unittest.TestCase):
   def test_equal(self):
      self.assertEqual(verif.axis.No(), verif.axis.No())
      self.assertTrue(verif.axis.No() == verif.axis.No())
      self.assertFalse(verif.axis.No() != verif.axis.No())

   def test_unequal(self):
      self.assertFalse(verif.axis.No() == verif.axis.Location())
      self.assertFalse(verif.axis.No() == verif.axis.Time())
      self.assertTrue(verif.axis.No() != verif.axis.Location())
      self.assertTrue(verif.axis.No() != verif.axis.Time())


if __name__ == '__main__':
   unittest.main()
