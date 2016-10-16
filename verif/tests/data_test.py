import unittest
import verif.data
import numpy as np


class TestData(unittest.TestCase):
   def test_doesnotexist(self):
      with self.assertRaises(SystemExit):
         data = verif.data.Data(filenames=["verif/tests/file1.txt", "verif/tests/doesnotexist.txt"])

   def test_simple(self):
      data = verif.data.Data(filenames=["verif/tests/file1.txt", "verif/tests/file2.txt"])
      lats = data.get_lats()
      self.assertTrue(len(lats) == 1)
      self.assertTrue(lats[0] == 42)
      lons = data.get_lons()
      self.assertTrue(len(lons) == 1)
      self.assertTrue(lons[0] == 23)

      offsets = data.get_axis_values("offset")
      self.assertEqual(2, offsets.shape[0])
      self.assertEqual(0, offsets[0])
      self.assertEqual(12, offsets[1])

      data.set_axis("date")
      dates = data.get_axis_values()
      self.assertEqual(3, dates.shape[0])
      data.set_file_index(0)
      # 20120101: Common offsets: 0 and 12, locations 41
      data.set_index(0)
      [obs, fcst] = data.get_scores(["obs", "fcst"])
      self.assertEqual(5, fcst[0])  # Offset 0
      self.assertEqual(7, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102: (missing obs at offset 12)
      data.set_index(1)
      [obs, fcst] = data.get_scores(["obs", "fcst"])
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(6, fcst[0])
      self.assertEqual(1, obs[0])
      data.set_file_index(1)
      # 20120101
      data.set_index(0)
      [obs, fcst] = data.get_scores(["obs", "fcst"])
      self.assertEqual(0, fcst[0])  # Offset 0
      self.assertEqual(1, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102
      data.set_index(1)
      [obs, fcst] = data.get_scores(["obs", "fcst"])
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(5, fcst[0])
      self.assertEqual(1, obs[0])


if __name__ == '__main__':
   unittest.main()
