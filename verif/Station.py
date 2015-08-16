class Station:
   def __init__(self, id, lat, lon, elev):
      self._id = id
      self._lat = lat
      self._lon = lon
      self._elev = elev
   def id(self, value=None):
      if value == None:
         return self._id
      else:
         self._id = value
   def lat(self, value=None):
      if value == None:
         return self._lat
      else:
         self._lat = value
   def lon(self, value=None):
      if value == None:
         return self._lon
      else:
         self._lon = value
   def elev(self, value=None):
      if value == None:
         return self._elev
      else:
         self._elev = value

   def __eq__(self, other):
      # TODO: Should the ids be checked?
      return self.lat() == other.lat() and self.lon() == other.lon() and self.elev() == other.elev()

   def __hash__(self):
      return hash((self.lat(), self.lon(), self.elev()))
