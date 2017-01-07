import unittest
import verif.axis
import verif.data
import verif.metric
import verif.output
import numpy as np


class TestOutput(unittest.TestCase):
   def test_text(self):
      inputs = [verif.input.get_input("verif/tests/files/file_sigdig.txt")]
      data = verif.data.Data(inputs=inputs)
      metric = verif.metric.Obs()

      output = verif.output.Standard(metric)
      output.text(data)


class TestStandard(unittest.TestCase):
   def _get(self):
      inputs = [verif.input.get_input(file) for file in ["examples/raw.txt", "examples/kf.txt"]]
      data = verif.data.Data(inputs=inputs)
      metric = verif.metric.Mae()
      output = verif.output.Standard(metric)
      return data, output

   def test_leadtime(self):
      data, output = self._get()
      x, y, _, _ = output._get_x_y(data, verif.axis.Leadtime())
      lt = data.leadtimes
      np.testing.assert_array_equal(x, lt)
      self.assertEqual(2, y.shape[1])
      self.assertEqual(len(lt), y.shape[0])

   def test_location(self):
      data, output = self._get()
      for ax in [verif.axis.Location(), verif.axis.Lat(), verif.axis.Lon(), verif.axis.Elev()]:
         x, y, _, _ = output._get_x_y(data, verif.axis.Location())
         loc = data.locations
         self.assertEqual(len(loc), len(x))
         self.assertEqual(2, y.shape[1])
         self.assertEqual(len(loc), y.shape[0])

   def test_year(self):
      data, output = self._get()
      x, y, _, _ = output._get_x_y(data, verif.axis.Year())
      # There should only be one year in this dataset
      self.assertEqual(1, len(x))
      self.assertEqual(1, y.shape[0])
      # The date represents the first day of the year:
      self.assertEqual(20120101, verif.util.datenum_to_date(x[0]))
      self.assertEqual(2, y.shape[1])

   def test_month(self):
      data, output = self._get()
      x, y, _, _ = output._get_x_y(data, verif.axis.Month())
      # There should be three months in this dataset
      self.assertEqual(3, len(x))
      self.assertEqual(3, y.shape[0])
      # The dates represents the first day of the months:
      self.assertEqual(20120101, verif.util.datenum_to_date(x[0]))
      self.assertEqual(20120201, verif.util.datenum_to_date(x[1]))
      self.assertEqual(20120301, verif.util.datenum_to_date(x[2]))
      self.assertEqual(2, y.shape[1])

   def test_time(self):
      data, output = self._get()
      x, y, _, _ = output._get_x_y(data, verif.axis.Time())
      # 20120101 to 20120301
      self.assertEqual(61, len(x))
      self.assertEqual(61, y.shape[0])
      self.assertEqual(20120101, verif.util.datenum_to_date(x[0]))
      self.assertEqual(20120102, verif.util.datenum_to_date(x[1]))
      self.assertEqual(20120301, verif.util.datenum_to_date(x[-1]))
      self.assertEqual(2, y.shape[1])

   def test_no(self):
      # Collapse all data into one number
      data, output = self._get()
      x, y, _, _ = output._get_x_y(data, verif.axis.No())
      self.assertEqual(1, len(x))
      self.assertEqual(1, y.shape[0])
      self.assertEqual(2, y.shape[1])


if __name__ == '__main__':
   unittest.main()
