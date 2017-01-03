import unittest
import verif.driver
import os
import numpy as np
np.seterr('raise')


class IntegrationTest(unittest.TestCase):
   # Runs a verif command line
   @staticmethod
   def run_command(command):
      argv = command.split()
      verif.driver.run(argv)

   @staticmethod
   def remove(file):
      os.remove(file)

   # In bytes
   @staticmethod
   def file_size(filename):
      statinfo = os.stat(filename)
      return statinfo.st_size

   @staticmethod
   def is_valid_image(filename):
      return IntegrationTest.file_size(filename) > 3000

   # Runs command and appends -f test.png
   # Cleans up the image file afterwards
   def run_with_image(self, command):
      imageFile = "test.png"
      command = command + " -f " + imageFile
      self.run_command(command)
      self.assertTrue(self.is_valid_image(imageFile))
      self.remove(imageFile)

   def test_valid(self):
      self.run_command("verif")
      self.run_command("verif --version")
      self.run_command("verif examples/raw.txt examples/kf.txt --list-thresholds")
      self.run_command("verif examples/raw.txt examples/kf.txt --list-quantiles")
      self.run_command("verif examples/raw.txt examples/kf.txt --list-thresholds --list-quantiles")

   def test_invalid(self):
      with self.assertRaises(SystemExit):
         self.run_with_image("verif --list-thresholds")
      with self.assertRaises(SystemExit):
         self.run_with_image("verif --list-quantiles")

   def test_README(self):
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m ets")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m taylor")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m error")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m reliability -r 0")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m pithist")

   def test_option_b(self):
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m ets -b below")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m ets -b within")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m ets -b above")

   def test_option_c(self):
      self.run_with_image("verif examples/raw.txt -c examples/kf.txt -m ets")

   def test_option_leg(self):
      self.run_with_image("verif -leg 1,2 examples/raw.txt examples/kf.txt -m ets")
      self.run_with_image("verif -leg 1,2 examples/raw.txt examples/kf.txt -m ets -x no")
      self.run_with_image("verif -leg 1dqwoijdioqwjdoiqjwdoijiqow,2dqwoijdioqwjdoiqjwdoijiqow examples/raw.txt examples/kf.txt -m ets")
      with self.assertRaises(SystemExit):
         self.run_with_image("verif -leg 1 examples/raw.txt examples/kf.txt -m ets")

   def test_option_ct(self):
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg min")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg mean")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg median")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg max")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg std")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -agg range")

   def test_option_x(self):
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x no")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x location")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x lat")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x lon")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x elev")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x leadtime")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x time")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x week")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x month")
      self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x year")

   def test_freq(self):
      self.run_with_image("verif verif/tests/files/file1.txt -m freq")
      # Check that no error occurs, even though fcst or obs is not available
      self.run_with_image("verif verif/tests/files/file1_no_obs.txt -m freq")
      self.run_with_image("verif verif/tests/files/file1_no_fcst.txt -m freq")

   def test_option_lc(self):
       for lc in ("g,r", "g", "g,r,b", "0,0.5,0.9", "[0,0,1],0.5,g"):
          self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lc %s" % lc)

   def test_options(self):
      for opt in ("acc", "nogrid", "nomargin", "hist", "sort", "sp", "simple", "xlog", "ylog"):
         self.run_with_image("verif examples/raw.txt examples/kf.txt -m obs -%s" % opt)

   def test_invalidMetric(self):
      with self.assertRaises(SystemExit):
         self.run_with_image("verif examples/T_raw_0.nc -m maeq")

   def test_invalidFile(self):
      with self.assertRaises(SystemExit):
         self.run_with_image("verif examples/T_raw_1.nc -m mae")


if __name__ == '__main__':
   unittest.main()
