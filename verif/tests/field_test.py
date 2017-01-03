import unittest
import verif.field
import numpy as np


class TestField(unittest.TestCase):
   def test_equal(self):
      self.assertEqual(verif.field.Obs(), verif.field.Obs())
      self.assertTrue(verif.field.Obs() == verif.field.Obs())
      self.assertFalse(verif.field.Obs() != verif.field.Obs())

      self.assertEqual(verif.field.Threshold(0), verif.field.Threshold(0))
      self.assertTrue(verif.field.Threshold(0) == verif.field.Threshold(0))
      self.assertFalse(verif.field.Threshold(0) != verif.field.Threshold(0))

   def test_unequal(self):
      self.assertFalse(verif.field.Obs() == verif.field.Fcst())
      self.assertFalse(verif.field.Obs() == verif.field.Threshold(0))
      self.assertTrue(verif.field.Obs() != verif.field.Fcst())
      self.assertTrue(verif.field.Obs() != verif.field.Threshold(0))

      self.assertFalse(verif.field.Threshold(0) == verif.field.Threshold(1))
      self.assertTrue(verif.field.Threshold(0) != verif.field.Threshold(1))


if __name__ == '__main__':
   unittest.main()
