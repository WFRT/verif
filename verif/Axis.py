class Axis:
   def isLocationAxis():
      return False
   pass


class Date:
   pass


class Offset:
   pass


class Station:
   def isLocationAxis():
      return True


class Lat:
   def isLocationAxis():
      return True


class Lon:
   def isLocationAxis():
      return True


class StationId:
   def isLocationAxis():
      return True


class Elev:
   def isLocationAxis():
      return True


class None:
   pass


class All:
   pass


class Threshold:
   pass
