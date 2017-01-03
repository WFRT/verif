import unittest
import numpy as np
from verif.location import Location


class TestLocation(unittest.TestCase):
   def test_equal(self):
      self.assertEqual(Location(0,1,2,3), Location(0,1,2,3))
      self.assertTrue(Location(0,1,2,3) == Location(0,1,2,3))
      self.assertFalse(Location(0,1,2,3) != Location(0,1,2,3))

   def test_unequal(self):
      for location in [Location(0,1,2,2), Location(0,1,3,3), Location(0,2,2,3), Location(1,1,2,3)]:
         self.assertFalse(Location(0,1,2,3) == location)
         self.assertFalse(Location(0,1,2,3) == location)
         self.assertTrue(Location(0,1,2,3) != location)
         self.assertTrue(Location(0,1,2,3) != location)

   def test_distance(self):
      loc1 = Location(0, lat=60, lon=10, elev=0)
      loc2 = Location(0, lat=-30, lon=-40, elev=100)
      self.assertEqual(0, loc1.get_distance(loc1))
      # Computed using: http://www.onlineconversion.com/map_greatcircle_distance.htm
      self.assertAlmostEqual(10996966.188406856, loc1.get_distance(loc2))
      self.assertEqual(loc2.get_distance(loc1), loc1.get_distance(loc2))


if __name__ == '__main__':
   unittest.main()
