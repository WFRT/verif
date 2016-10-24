Date = 1
Offset = 2
Location = 3
Lat = 31
Lon = 32
Elev = 33
LocationId = 34
All = 4
No = 5
Week = 6
Month = 7
Year = 8
DayOfMonth = 9
DayOfYear = 10
Threshold = 11

def get_axis(name):
   if name == "date":
      return Date
   elif name == "offset":
      return Offset
   elif name == "location":
      return Location
   elif name == "lat":
      return Lat
   elif name == "lon":
      return Lon
   elif name == "elev":
      return Elev
   elif name == "locationId":
      return LocationId
   elif name == "all":
      return All
   elif name == "none":
      return No
   elif name == "week":
      return Week
   elif name == "month":
      return Month
   elif name == "threshold":
      return Threshold
   else:
      abort()
