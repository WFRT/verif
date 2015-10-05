import unittest
import verif.Data as Data
import verif.Util as Util
import numpy as np


class TestData(unittest.TestCase):
   def test_doesnotexist(self):
      with self.assertRaises(SystemExit):
         data = Data.Data(filenames=["tests/file1.txt", "doesnotexist.txt"])

   def test_simple(self):
      data = Data.Data(filenames=["tests/file1.txt", "tests/file2.txt"])
      lats = data.getLats()
      self.assertTrue(len(lats) == 1)
      self.assertTrue(lats[0] == 42)
      lons = data.getLons()
      self.assertTrue(len(lons) == 1)
      self.assertTrue(lons[0] == 23)

      offsets = data.getAxisValues("offset")
      self.assertEqual(2, offsets.shape[0])
      self.assertEqual(0, offsets[0])
      self.assertEqual(12, offsets[1])

      data.setAxis("date")
      dates = data.getAxisValues()
      self.assertEqual(3, dates.shape[0])
      data.setFileIndex(0)
      # 20120101: Common offsets: 0 and 12, locations 41
      data.setIndex(0)
      [obs, fcst] = data.getScores(["obs", "fcst"])
      self.assertEqual(5, fcst[0])  # Offset 0
      self.assertEqual(7, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102: (missing obs at offset 12)
      data.setIndex(1)
      [obs, fcst] = data.getScores(["obs", "fcst"])
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(6, fcst[0])
      self.assertEqual(1, obs[0])
      data.setFileIndex(1)
      # 20120101
      data.setIndex(0)
      [obs, fcst] = data.getScores(["obs", "fcst"])
      self.assertEqual(0, fcst[0])  # Offset 0
      self.assertEqual(1, fcst[1])  # Offset 12
      self.assertEqual(-1, obs[0])
      self.assertEqual(2, obs[1])
      # 20120102
      data.setIndex(1)
      [obs, fcst] = data.getScores(["obs", "fcst"])
      self.assertEqual(1, fcst.shape[0])
      self.assertEqual(5, fcst[0])
      self.assertEqual(1, obs[0])


if __name__ == '__main__':
   unittest.main()
