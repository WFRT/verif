import unittest
import numpy as np
import verif.input
import verif.location
import verif.data


class InputNetcdfTest(unittest.TestCase):
   def test_valid2(self):
      input = verif.input.Netcdf("verif/tests/files/netcdf_valid2.nc")
      locations = input.locations
      self.assertTrue(1, len(locations))
      self.assertEqual(verif.location.Location(18700, 59.9423, 10.72, 94), locations[0])
      np.testing.assert_array_equal(np.array([0, 1, 2]), input.leadtimes)
      np.testing.assert_array_equal(np.array([1388534400, 1388620800]), input.times)
      obs = input.obs
      self.assertEqual(2, obs.shape[0])  # Time
      self.assertEqual(3, obs.shape[1])  # Leadtime
      self.assertEqual(1, obs.shape[2])  # Location
      self.assertAlmostEqual(1, obs[0, 0, 0])
      self.assertAlmostEqual(2, obs[0, 1, 0])
      self.assertAlmostEqual(3, obs[0, 2, 0])
      self.assertAlmostEqual(4, obs[1, 0, 0])
      self.assertAlmostEqual(5, obs[1, 1, 0])
      self.assertAlmostEqual(6, obs[1, 2, 0])
      self.assertEqual(0, len(input.thresholds))
      self.assertEqual(0, len(input.quantiles))
      self.assertTrue(input.ensemble is None)
      self.assertTrue(verif.field.Obs() in input.get_fields())
      self.assertFalse(verif.field.Fcst() in input.get_fields())

   def test_is_valid(self):
      self.assertTrue(verif.input.Netcdf.is_valid("verif/tests/files/netcdf_valid1.nc"))
      self.assertTrue(verif.input.Netcdf.is_valid("verif/tests/files/netcdf_valid2.nc"))

   def test_invalid(self):
      self.assertFalse(verif.input.Comps.is_valid("verif/tests/files/netcdf_invalid1.nc"))
      self.assertFalse(verif.input.Comps.is_valid("verif/tests/files/netcdf_invalid2.nc"))

if __name__ == '__main__':
   unittest.main()
