import numpy as np

class Location(object):
   def __init__(self, id, lat, lon, elev):
      self.id = id
      self.lat = lat
      self.lon = lon
      self.elev = elev

   def get_distance(self, other):
      lat1 = self.lat
      lon1 = self.lon
      lat2 = other.lat
      lon2 = other.lon

      def deg2rad(deg):
         return deg * np.pi / 180.0
      radiusEarth = 6.37e6
      lat1r = deg2rad(lat1)
      lat2r = deg2rad(lat2)
      lon1r = deg2rad(lon1)
      lon2r = deg2rad(lon2)
      ratio = np.cos(lat1r) * np.cos(lon1r) * np.cos(lat2r) * np.cos(lon2r) +\
              np.cos(lat1r) * np.sin(lon1r) * np.cos(lat2r) * np.sin(lon2r) +\
              np.sin(lat1r) * np.sin(lat2r)
      dist = np.arccos(ratio) * radiusEarth
      return dist

   def __eq__(self, other):
      # TODO: Should the ids be checked?
      return self.lat == other.lat and self.lon == other.lon and self.elev == other.elev

   def __hash__(self):
      return hash((self.lat, self.lon, self.elev))

   def __str__(self):
      return "(id,lat,lon,elev) = (%g,%g,%g,%g)" % (self.id, self.lat, self.lon, self.elev)
