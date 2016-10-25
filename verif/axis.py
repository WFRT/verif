import matplotlib.ticker
import verif.util


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
MonthOfYear = 12

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
   elif name == "year":
      return Year
   elif name == "dayofmonth":
      return DayOfMonth
   elif name == "dayofyear":
      return DayOfYear
   elif name == "threshold":
      return Threshold
   else:
      verif.util.error("Unknown axis: %d" % name)

def get_name(axis):
   if axis == verif.axis.Date:
      return "date"
   elif axis == verif.axis.Offset:
      return "offset"
   elif axis == verif.axis.Location:
      return "location"
   elif axis == verif.axis.Lat:
      return "lat"
   elif axis == verif.axis.Lon:
      return "lon"
   elif axis == verif.axis.Elev:
      return "elev"
   elif axis == verif.axis.LocationId:
      return "locationId"
   elif axis == verif.axis.All:
      return "all"
   elif axis == verif.axis.None:
      return "no"
   elif axis == verif.axis.Week:
      return "week"
   elif axis == verif.axis.Month:
      return "month"
   elif axis == verif.axis.Year:
      return "year"
   elif axis == verif.axis.Dayofmonth:
      return "dayOfMonth"
   elif axis == verif.axis.Dayofyear:
      return "dayOfYear"
   elif axis == verif.axis.Threshold:
      return "threshold"
   else:
      verif.util.error("Unknown axis: %d" % axis)

def is_continuous(axis):
   """
   Is this axis represent a notion that is continuous? I.e. does it make
   sense to connect dots with a line on a graph? For example, Location is
   not a continuous axis.
   """
   return axis in [verif.axis.Date, verif.axis.Offset, verif.axis.Threshold,
         verif.axis.Month, verif.axis.Year]

def is_date_like(axis):
   """ Does this axis have anything to do with the notion of date? """
   return axis in [verif.axis.Date, verif.axis.Month, verif.axis.Year]

def is_location_like(axis):
   if(axis is None):
      return False
   return axis in [verif.axis.Location, verif.axis.LocationId,
         verif.axis.Lat, verif.axis.Lon, verif.axis.Elev]

def get_formatter(axis):
   """ What format should this axis have? Returns an mpl Formatter """

   # Note the date formatters are never retrieved from here, since
   # mpl.gca().xaxis_date() is used instead
   if(axis == verif.axis.Date):
      return matplotlib.ticker.DateFormatter('\n%Y-%m-%d')
   elif(axis == verif.axis.Month):
      return matplotlib.ticker.DateFormatter('\n%Y-%m')
   elif(axis == verif.axis.Year):
      return matplotlib.ticker.DateFormatter('\n%Y')
   else:
      return matplotlib.ticker.ScalarFormatter()

