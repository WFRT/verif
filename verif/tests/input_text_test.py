import unittest
import numpy as np
import verif
import verif.input
import verif.data
import verif.location


class MyTest(unittest.TestCase):
   def test_get_leadtimes(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      leadtimes = input.leadtimes
      self.assertEqual(4, leadtimes.shape[0])
      self.assertEqual(0, leadtimes[0])
      self.assertEqual(1, leadtimes[1])
      self.assertEqual(3, leadtimes[2])
      self.assertEqual(22, leadtimes[3])

   def test_get_times(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      times = input.times
      self.assertEqual(1, times.shape[0])
      self.assertEqual(1325376000, times[0])

   def test_get_locations(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      locations = input.locations
      locations = np.sort(locations)
      self.assertEqual(4, len(locations))
      self.assertTrue(verif.location.Location(0, 1, 1, 1) in locations)
      self.assertTrue(verif.location.Location(0, 0, 0, 1) in locations)
      self.assertTrue(verif.location.Location(0, 0, 0, 2) in locations)
      self.assertTrue(verif.location.Location(0, 2, 2, 1) in locations)

   def test_conflictingLocations(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileConflictingInfo.txt")])

   def test_noId(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoId.txt")])

   def test_noElev(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoElev.txt")])
      self.assertEqual(2, len(data.locations))

   def test_noLat(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoLat.txt")])
      self.assertEqual(2, len(data.locations))

   def test_get_scores(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      obs = input.obs
      fcst = input.fcst
      self.assertEqual((1, 4, 4), obs.shape)
      self.assertEqual((1, 4, 4), fcst.shape)
      locations = input.locations
      I0 = -1
      for i in range(0, len(locations)):
         if(locations[i] == verif.location.Location(0, 1, 1, 1)):
            I0 = i
         if(locations[i] == verif.location.Location(0, 0, 0, 2)):
            I1 = i
      self.assertEqual(12, obs[0, 0, I0])
      self.assertTrue(np.isnan(obs[0, 1, I0]))
      self.assertTrue(np.isnan(obs[0, 2, I0]))
      self.assertEqual(5, obs[0, 3, I0])
      self.assertEqual(13, fcst[0, 0, I0])
      self.assertTrue(np.isnan(fcst[0, 1, I0]))
      self.assertTrue(np.isnan(fcst[0, 2, I0]))
      self.assertEqual(4, fcst[0, 3, I0])

      for i in range(0, obs.shape[1]):
         self.assertTrue(np.isnan(obs[0, i, I1]))
      self.assertEqual(11, fcst[0, 0, I1])
      for i in range(1, obs.shape[1]):
         self.assertTrue(np.isnan(obs[0, i, I1]))

if __name__ == '__main__':
   unittest.main()
