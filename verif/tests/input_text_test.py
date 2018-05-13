import unittest
import numpy as np
import verif
import verif.input
import verif.data
import verif.location


class InputTextTest(unittest.TestCase):
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

   def test_get_locations_no_id(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      locations = input.locations
      locations = np.sort(locations)
      # Reset ids, to make it easier to verify what locations are available
      for loc in locations:
         loc.id = 0
      self.assertEqual(4, len(locations))
      self.assertTrue(verif.location.Location(0, 1, 1, 1) in locations)
      self.assertTrue(verif.location.Location(0, 0, 0, 1) in locations)
      self.assertTrue(verif.location.Location(0, 0, 0, 2) in locations)
      self.assertTrue(verif.location.Location(0, 2, 2, 1) in locations)

   def test_get_locations(self):
      input = verif.input.Text("verif/tests/files/file1.txt")
      locations = input.locations
      locations = np.sort(locations)
      self.assertEqual(2, len(locations))
      self.assertTrue(verif.location.Location(3, 50, 10, 12) in locations)
      self.assertTrue(verif.location.Location(41, 42, 23, 341) in locations)

   def test_get_locations_same_lat_lon(self):
      input1 = verif.input.Text("verif/tests/files/file1.txt")
      input2 = verif.input.Text("verif/tests/files/file1_same_lat_lon.txt")
      locations1 = input1.locations
      locations2 = input2.locations
      self.assertEqual(len(locations1), len(locations2))

      # Find the ordering of locations, such that obs, fcsts can be compared
      # Create a list of indicies that map location1 into location 2
      I = list()
      for i in range(0, len(locations1)):
         for j in range(0, len(locations2)):
            if locations1[i].id == locations2[j].id:
               I += [j]

      self.assertEqual(len(locations1), len(I))
      locations2 = np.sort(locations2)
      self.assertEqual(2, len(locations2))
      self.assertTrue(verif.location.Location(3, 50, 10, 12) in locations2)
      self.assertTrue(verif.location.Location(41, 50, 10, 12) in locations2)

      np.testing.assert_array_equal(input1.obs[:, :, I], input2.obs)
      np.testing.assert_array_equal(input1.fcst[:, :, I], input2.fcst)
      np.testing.assert_array_equal(input1.leadtimes, input2.leadtimes)
      np.testing.assert_array_equal(input1.times, input2.times)

   def test_offset_leadtime(self):
      input1 = verif.input.Text("verif/tests/files/file1.txt")
      input2 = verif.input.Text("verif/tests/files/file1_leadtime.txt")
      np.testing.assert_array_equal(input1.obs, input2.obs)
      np.testing.assert_array_equal(input1.fcst, input2.fcst)
      np.testing.assert_array_equal(input1.leadtimes, input2.leadtimes)

   def test_conflictingLocations(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileConflictingInfo.txt")])

   def test_noId(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoLocation.txt")])

   def test_noElev(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoElev.txt")])
      self.assertEqual(2, len(data.locations))

   def test_noLat(self):
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/fileNoLat.txt")])
      self.assertEqual(2, len(data.locations))

   def test_odd_header_names(self):
      # Check that names 'q', 'p', and 'e' can be read
      data = verif.data.Data(inputs=[verif.input.Text("verif/tests/files/file_odd_header_names.txt")])

   def test_hour(self):
      # Check that the hour column works
      input = verif.input.Text("verif/tests/files/text_hour.txt")
      self.assertEqual((3, 2, 1), input.obs.shape)
      np.testing.assert_array_equal(input.leadtimes, [0, 2])
      ut = verif.util.date_to_unixtime(20180101)
      np.testing.assert_array_equal(input.times, [ut, ut + 12 * 3600, ut + 36 * 3600])
      self.assertTrue(np.isnan(input.obs[2, 0, 0]))
      self.assertEqual(9, input.obs[2, 1, 0])

   def test_get_scores(self):
      input = verif.input.Text("verif/tests/files/example.txt")
      obs = input.obs
      fcst = input.fcst
      self.assertEqual((1, 4, 4), obs.shape)
      self.assertEqual((1, 4, 4), fcst.shape)
      locations = input.locations
      self.assertEqual(4, len(locations))
      # Reset ids, to make it easier to verify what locations are available
      for loc in locations:
         loc.id = 0
      I0 = -1
      for i in range(0, len(locations)):
         if locations[i] == verif.location.Location(0, 1, 1, 1):
            I0 = i
         if locations[i] == verif.location.Location(0, 0, 0, 2):
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

   def test_compatibility(self):
      input = verif.input.Text("verif/tests/files/file1_compatibility.txt")
      locations = input.locations
      locations = np.sort(locations)
      self.assertEqual(2, len(locations))
      self.assertTrue(verif.location.Location(3, 50, 10, 12) in locations)
      self.assertTrue(verif.location.Location(41, 42, 23, 341) in locations)
      np.testing.assert_array_equal(np.sort(input.leadtimes), [0, 6, 12])

   def test_order(self):
      # Check that the order of leadtimes in the file does not matter
      input1 = verif.input.Text("verif/tests/files/file_order1.txt")
      input2 = verif.input.Text("verif/tests/files/file_order2.txt")
      data = verif.data.Data([input1, input2])
      np.testing.assert_array_equal(np.sort(input1.leadtimes), np.sort(input2.leadtimes))
      np.testing.assert_array_equal(np.sort(data.leadtimes), [0, 6, 12])
      s1 = data.get_scores(verif.field.Fcst(), 0)
      s2 = data.get_scores(verif.field.Fcst(), 1)
      np.testing.assert_array_equal(s1, s2)

   def test_attributes(self):
      input = verif.input.Text("verif/tests/files/file1_x0_x1.txt")
      data = verif.data.Data([input])
      self.assertEqual(0, data.variable.x0)
      self.assertEqual(10, data.variable.x1)
      self.assertEqual("Weird variable", data.variable.name)
      self.assertEqual("Some units", data.variable.units)

   def test_missing_header(self):
      with self.assertRaises(SystemExit) as context:
         input = verif.input.Text("verif/tests/files/file1_no_header.txt")

   def test_missing_values(self):
      # Check NA, na, nan are detected
      input = verif.input.Text("verif/tests/files/file_missing_values.txt")
      data = verif.data.Data([input])
      obs = data.get_scores(verif.field.Obs(), 0)
      fcst = data.get_scores(verif.field.Fcst(), 0)
      self.assertEqual(len(data.locations), 1)

      self.assertEqual(obs[0, 0, 0], 3)
      self.assertEqual(fcst[0, 0, 0], 6)

      self.assertTrue(np.isnan(obs[0, 2, 0]))
      self.assertEqual(fcst[0, 2, 0], 7)

      self.assertEqual(obs[0, 3, 0], 3)
      self.assertTrue(np.isnan(fcst[0, 3, 0]))

      self.assertTrue(np.isnan(obs[0, 4, 0]))
      self.assertTrue(np.isnan(fcst[0, 4, 0]))

if __name__ == '__main__':
   unittest.main()
