class Station:
   def __init__(self, id, lat, lon, elev):
      self._id = id
      self._lat = lat
      self._lon = lon
      self._elev = elev
   def id(self):
      return self._id
   def lat(self):
      return self._lat
   def lon(self):
      return self._lon
   def elev(self):
      return self._elev

