import unittest
import verif.data
import numpy as np


def get_data_from_text(filename):
   data = verif.data.Data(verif.input.Text(filename))
   return data


def assert_set_equal(array1, array2):
   np.testing.assert_array_equal(np.sort(array1), np.sort(array2))


class TestData(unittest.TestCase):
   def test_doesnotexist(self):
      with self.assertRaises(SystemExit):
         inputs = [verif.input.get_input(filename) for filename in ["verif/tests/files/file1.txt",
            "verif/tests/files/doesnotexist.txt"]]
         data = verif.data.Data(inputs=inputs)

   def test_simple(self):
      inputs = [verif.input.get_input(filename) for filename in ["verif/tests/files/file1.txt",
         "verif/tests/files/file2.txt"]]
      data = verif.data.Data(inputs=inputs)
      lats = [loc.lat for loc in data.locations]
      self.assertTrue(len(lats) == 1)
      self.assertTrue(lats[0] == 42)
      lons = [loc.lon for loc in data.locations]
      self.assertTrue(len(lons) == 1)
      self.assertTrue(lons[0] == 23)

      leadtimes = data.leadtimes
      self.assertEqual(2, leadtimes.shape[0])
      self.assertEqual(0, leadtimes[0])
      self.assertEqual(12, leadtimes[1])

      times = data.times
      self.assertEqual(3, times.shape[0])
      # 20120101: Common leadtimes: 0 and 12, locations 41
      fields = [verif.field.Obs(), verif.field.Fcst()]
      axis = verif.axis.Time()

      [obs, fcst] = data.get_scores(fields, 0, axis, 0)
      self.assertEqual(5, fcst[0])  # Offset 0
      self.assertEqual(7, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102: (missing obs at leadtime 12)
      [obs, fcst] = data.get_scores(fields, 0, axis, 1)
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(6, fcst[0])
      self.assertEqual(1, obs[0])

      # 20120101
      [obs, fcst] = data.get_scores(fields, 1, axis, 0)
      self.assertEqual(0, fcst[0])  # Offset 0
      self.assertEqual(1, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102
      [obs, fcst] = data.get_scores(fields, 1, axis, 1)
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(5, fcst[0])
      self.assertEqual(1, obs[0])

   def test_latrange(self):
      inputs = [verif.input.Text("verif/tests/files/file1.txt"), verif.input.Text("verif/tests/files/file3.txt")]
      data = verif.data.Data(inputs, lat_range=[44, 60])  # Only 1 common station within the range
      self.assertEqual(1, len(data.locations))
      self.assertEqual(50, data.locations[0].lat)
      self.assertEqual(10, data.locations[0].lon)
      data = verif.data.Data(inputs, lat_range=[40, 60])
      self.assertEqual(2, len(data.locations))
      with self.assertRaises(SystemExit):
         data = verif.data.Data(inputs, lat_range=[55, 60])

   def test_latrange_locations(self):
      # Check that the intersection of lat_range and locations is correct
      inputs = [verif.input.Text("verif/tests/files/file_many_locations.txt")]
      data = verif.data.Data(inputs, lat_range=[25, 60], locations=[3, 2, 21])
      self.assertEqual(2, len(data.locations))
      self.assertTrue(3 in [loc.id for loc in data.locations])
      self.assertTrue(21 in [loc.id for loc in data.locations])

   def test_lonrange_locations(self):
      # Check that the intersection of lon_range and locations is correct
      inputs = [verif.input.Text("verif/tests/files/file_many_locations.txt")]
      data = verif.data.Data(inputs, lon_range=[6, 16], locations=[1, 2, 911])
      self.assertEqual(2, len(data.locations))
      self.assertTrue(1 in [loc.id for loc in data.locations])
      self.assertTrue(2 in [loc.id for loc in data.locations])

   def test_dayofyear(self):
      inputs = [verif.input.get_input("verif/tests/files/file1.txt")]
      data = verif.data.Data(inputs=inputs)
      axis = verif.axis.Dayofyear()
      axis_values = data.get_axis_values(axis)
      self.assertEqual(3, len(axis_values))
      self.assertEqual(1, axis_values[0])
      self.assertEqual(2, axis_values[1])
      self.assertEqual(3, axis_values[2])

      # All the values for day 1
      values = data.get_scores([verif.field.Fcst()], 0, axis, 0)[0]
      self.assertEqual(6, len(values))
      assert_set_equal(np.array([6, 7, 7, 5, 4, 7]), values)

      values = data.get_scores([verif.field.Fcst()], 0, axis, 2)[0]
      self.assertEqual(5, len(values))
      assert_set_equal(np.array([-4, 3, 9, 12, 16]), values)


class TestDataRemovingLocations(unittest.TestCase):
   def test_inside_lat_range(self):
      inputs = [verif.input.Text("verif/tests/files/file3locations.txt")]
      data = verif.data.Data(inputs, lat_range=[39, 51], locations_x=[2])
      self.assertEqual(1, len(data.locations))
      self.assertEqual(1, data.locations[0].id)
      self.assertEqual(40, data.locations[0].lat)

   def test_outside_lat_range(self):
      inputs = [verif.input.Text("verif/tests/files/file3locations.txt")]
      data = verif.data.Data(inputs, lat_range=[39, 51], locations_x=[3])
      self.assertEqual(2, len(data.locations))
      self.assertTrue(1 in [loc.id for loc in data.locations])
      self.assertTrue(2 in [loc.id for loc in data.locations])

   def test_with_l(self):
      inputs = [verif.input.Text("verif/tests/files/file3locations.txt")]
      data = verif.data.Data(inputs, locations=[1, 2], locations_x=[2])
      self.assertEqual(1, len(data.locations))
      self.assertTrue(1 in [loc.id for loc in data.locations])

   def test_with_l(self):
      inputs = [verif.input.Text("verif/tests/files/file3locations.txt")]
      data = verif.data.Data(inputs, locations=[1, 2], locations_x=[3])
      self.assertEqual(2, len(data.locations))
      self.assertTrue(1 in [loc.id for loc in data.locations])
      self.assertTrue(2 in [loc.id for loc in data.locations])


class TestDataClim(unittest.TestCase):
   def test_names(self):
      """
      Checks that climatology files do not end up in get_names()
      and related functions
      """
      inputs = [verif.input.Text("verif/tests/files/file1.txt")]
      clim = verif.input.get_input("verif/tests/files/file2.txt")
      data = verif.data.Data(inputs, clim=clim)
      self.assertEqual(1, len(data.get_names()))
      self.assertEqual(1, len(data.get_short_names()))
      self.assertEqual(1, len(data.get_full_names()))


class TestDataFields(unittest.TestCase):
   def test_get_fields1(self):
      inputs = [verif.input.Text("verif/tests/files/file1.txt")]
      data = verif.data.Data(inputs)
      self.assertTrue(verif.field.Fcst() in data.get_fields())
      self.assertTrue(verif.field.Obs() in data.get_fields())

   def test_get_fields2(self):
      inputs = [verif.input.Text("verif/tests/files/file1_no_fcst.txt")]
      data = verif.data.Data(inputs)
      self.assertFalse(verif.field.Fcst() in data.get_fields())
      self.assertTrue(verif.field.Obs() in data.get_fields())

   def test_get_fields3(self):
      # Only Obs is common between the two files
      inputs = [verif.input.Text("verif/tests/files/%s" % file) for file in ["file1.txt", "file1_no_fcst.txt"]]
      data = verif.data.Data(inputs)
      self.assertFalse(verif.field.Fcst() in data.get_fields())
      self.assertTrue(verif.field.Obs() in data.get_fields())


if __name__ == '__main__':
   unittest.main()
