import unittest
import verif.driver
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
      return IntegrationTest.file_size(filename) > 10000

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

   # def test_option_hist(self):
   #    self.run_with_image("verif examples/raw.txt examples/kf.txt -m obs -hist")

   def test_invalidMetric(self):
      with self.assertRaises(SystemExit):
         self.run_with_image("verif examples/T_raw_0.nc -m maeq")

   def test_invalidFile(self):
      with self.assertRaises(SystemExit):
         self.run_with_image("verif examples/T_raw_1.nc -m mae")
