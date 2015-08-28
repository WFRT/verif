import matplotlib
matplotlib.use('Agg')
import unittest
import verif.Driver as Driver
import verif.Common as Common
import os
import numpy as np
np.seterr('raise')

class IntegrationTest(unittest.TestCase):
   # Runs a verif command line
   @staticmethod
   def runCommand(command):
      argv = command.split()
      Driver.run(argv)

   @staticmethod
   def remove(file):
      os.remove(file)

   # In bytes
   @staticmethod
   def fileSize(filename):
      statinfo = os.stat(filename)
      return statinfo.st_size

   @staticmethod
   def isValidImage(filename):
      return IntegrationTest.fileSize(filename) > 10000

   # Runs command and appends -f test.png
   # Cleans up the image file afterwards
   def runWithImage(self, command):
      imageFile = "test.png"
      command = command + " -f " + imageFile
      self.runCommand(command)
      self.assertTrue(self.isValidImage(imageFile))
      self.remove(imageFile)

   def test_valid(self):
      self.runCommand("verif")
      self.runCommand("verif --version")

   def test_README(self):
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m ets")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m taylor")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m error")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m reliability -r 0")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m pithist")

   def test_option_b(self):
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m ets -b below")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m ets -b within")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m ets -b above")

   def test_option_c(self):
      self.runWithImage("verif examples/raw.txt -c examples/kf.txt -m ets")

   def test_option_ct(self):
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct min")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct mean")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct median")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct max")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct std")
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m mae -ct range")

   def test_option_hist(self):
      self.runWithImage("verif examples/raw.txt examples/kf.txt -m obs -hist")

   def test_invalidMetric(self):
      with self.assertRaises(SystemExit):
         self.runWithImage("verif examples/T_raw_0.nc -m maeq")

   def test_invalidFile(self):
      with self.assertRaises(SystemExit):
         self.runWithImage("verif examples/T_raw_1.nc -m mae")
