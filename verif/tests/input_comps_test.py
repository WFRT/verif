import unittest
import numpy as np
import verif.input
import verif.location
import verif.data


class MyTest(unittest.TestCase):
   def test_comps_to_verif_threshold(self):
      self.assertEqual(0, verif.input.Comps._comps_to_verif_threshold("p0"))
      self.assertEqual(0.1, verif.input.Comps._comps_to_verif_threshold("p01"))
      self.assertEqual(0.01, verif.input.Comps._comps_to_verif_threshold("p001"))
      self.assertEqual(2, verif.input.Comps._comps_to_verif_threshold("p2"))
      self.assertEqual(1.1, verif.input.Comps._comps_to_verif_threshold("p1.1"))
      self.assertEqual(25.1, verif.input.Comps._comps_to_verif_threshold("p25.1"))
      self.assertEqual(-2, verif.input.Comps._comps_to_verif_threshold("pm2"))
      self.assertEqual(-1.2, verif.input.Comps._comps_to_verif_threshold("pm1.2"))

      self.assertEqual(None, verif.input.Comps._comps_to_verif_threshold("qwoei"))
      self.assertEqual(None, verif.input.Comps._comps_to_verif_threshold("q0.r"))

   def test_comps_to_verif_quantile(self):
      self.assertEqual(0, verif.input.Comps._comps_to_verif_quantile("q0"))
      self.assertEqual(0.1, verif.input.Comps._comps_to_verif_quantile("q10"))
      self.assertEqual(0.01, verif.input.Comps._comps_to_verif_quantile("q1"))
      self.assertEqual(0.001, verif.input.Comps._comps_to_verif_quantile("q01"))
      self.assertEqual(1, verif.input.Comps._comps_to_verif_quantile("q100"))

      self.assertEqual(None, verif.input.Comps._comps_to_verif_quantile("q101"))
      self.assertEqual(None, verif.input.Comps._comps_to_verif_quantile("qm101"))

   def test_verif_to_comps_threshold(self):
      self.assertEqual("p0", verif.input.Comps._verif_to_comps_threshold(0))
      self.assertEqual("p01", verif.input.Comps._verif_to_comps_threshold(0.1))
      self.assertEqual("p1", verif.input.Comps._verif_to_comps_threshold(1))
      self.assertEqual("p10", verif.input.Comps._verif_to_comps_threshold(10))
      self.assertEqual("pm01", verif.input.Comps._verif_to_comps_threshold(-0.1))
      self.assertEqual("pm1", verif.input.Comps._verif_to_comps_threshold(-1))
      self.assertEqual("pm10", verif.input.Comps._verif_to_comps_threshold(-10))

   def test_verif_to_comps_quantile(self):
      self.assertEqual("q0", verif.input.Comps._verif_to_comps_quantile(0))
      self.assertEqual("q10", verif.input.Comps._verif_to_comps_quantile(0.1))
      self.assertEqual("q1", verif.input.Comps._verif_to_comps_quantile(0.01))
      self.assertEqual("q01", verif.input.Comps._verif_to_comps_quantile(0.001))
      self.assertEqual("q100", verif.input.Comps._verif_to_comps_quantile(1))

      self.assertEqual(None, verif.input.Comps._verif_to_comps_quantile(-1))
      self.assertEqual(None, verif.input.Comps._verif_to_comps_quantile(2))

   def test_valid(self):
      self.assertTrue(verif.input.Comps.is_valid("verif/tests/files/comps_valid1.nc"))
      self.assertTrue(verif.input.Comps.is_valid("verif/tests/files/comps_valid2.nc"))

   def test_invalid(self):
      self.assertFalse(verif.input.Comps.is_valid("verif/tests/files/comps_invalid1.nc"))

if __name__ == '__main__':
   unittest.main()
