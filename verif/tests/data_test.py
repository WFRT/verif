import unittest
import verif.data
import numpy as np


def get_data_from_text(filename):
   data = verif.data.Data(verif.input.Text(filename))
   return data


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

      offsets = data.offsets
      self.assertEqual(2, offsets.shape[0])
      self.assertEqual(0, offsets[0])
      self.assertEqual(12, offsets[1])

      times = data.times
      self.assertEqual(3, times.shape[0])
      # 20120101: Common offsets: 0 and 12, locations 41
      fields = [verif.field.Obs(), verif.field.Fcst()]
      axis = verif.axis.Time()

      [obs, fcst] = data.get_scores(fields, 0, axis, 0)
      self.assertEqual(5, fcst[0])  # Offset 0
      self.assertEqual(7, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102: (missing obs at offset 12)
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
      data = verif.data.Data(inputs, lat_range=[44, 60]) # Only 1 common station within the range
      self.assertEqual(1, len(data.locations))
      self.assertEqual(50, data.locations[0].lat)
      self.assertEqual(10, data.locations[0].lon)
      data = verif.data.Data(inputs, lat_range=[40, 60])
      self.assertEqual(2, len(data.locations))
      with self.assertRaises(SystemExit):
         data = verif.data.Data(inputs, lat_range=[55, 60])
         

if __name__ == '__main__':
   unittest.main()
