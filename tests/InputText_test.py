import unittest
import verif.Input as Input
import verif.Station as Station
import numpy as np

class MyTest(unittest.TestCase):
   def test_getOffsets(self):
      input = Input.Text("tests/example.txt")
      offsets = input.getOffsets()
      self.assertEqual(4, offsets.shape[0])
      self.assertEqual(0, offsets[0])
      self.assertEqual(1, offsets[1])
      self.assertEqual(3, offsets[2])
      self.assertEqual(22, offsets[3])
   def test_getDates(self):
      input = Input.Text("tests/example.txt")
      dates = input.getDates()
      self.assertEqual(1, dates.shape[0])
      self.assertEqual(20120101, dates[0])
   def test_getStations(self):
      input = Input.Text("tests/example.txt")
      stations = input.getStations()
      stations = np.sort(stations)
      self.assertEqual(4, len(stations))
      self.assertTrue(Station.Station(0,1,1,1) in stations)
      self.assertTrue(Station.Station(0,0,0,1) in stations)
      self.assertTrue(Station.Station(0,0,0,2) in stations)
      self.assertTrue(Station.Station(0,2,2,1) in stations)
   def test_getScores(self):
      input = Input.Text("tests/example.txt")
      obs  = input.getScores("obs")
      fcst = input.getScores("fcst")
      self.assertEqual((1,4,4), obs.shape)
      self.assertEqual((1,4,4), fcst.shape)
      stations = input.getStations()
      I0 = -1
      for i in range(0, len(stations)):
         if(stations[i] == Station.Station(0,1,1,1)):
            I0 = i
         if(stations[i] == Station.Station(0,0,0,2)):
            I1 = i
      self.assertEqual(12, obs[0,0,I0])
      self.assertTrue(np.isnan(obs[0,1,I0]))
      self.assertTrue(np.isnan(obs[0,2,I0]))
      self.assertEqual(5, obs[0,3,I0])
      self.assertEqual(13, fcst[0,0,I0])
      self.assertTrue(np.isnan(fcst[0,1,I0]))
      self.assertTrue(np.isnan(fcst[0,2,I0]))
      self.assertEqual(4, fcst[0,3,I0])

      for i in range(0, obs.shape[1]):
         self.assertTrue(np.isnan(obs[0,i,I1]))
      self.assertEqual(11, fcst[0,0,I1])
      for i in range(1, obs.shape[1]):
         self.assertTrue(np.isnan(obs[0,i,I1]))

if __name__ == '__main__':
   unittest.main()
