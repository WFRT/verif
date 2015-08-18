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
      return IntegrationTest.fileSize(filename) > 30000

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
      self.runWithImage("verif examples/T_raw_0.nc -m mae")
      self.runWithImage("verif examples/T_raw_0.nc -m bias")
      self.runWithImage("verif examples/T_raw_0.nc -m rmse")
      self.runWithImage("verif examples/T_raw_0.nc -m reliability -r 0")

   def test_invalidMetric(self):
      with self.assertRaises(SystemExit):
         self.runWithImage("verif examples/T_raw_0.nc -m maeq")

   def test_invalidFile(self):
      with self.assertRaises(SystemExit):
         self.runWithImage("verif examples/T_raw_1.nc -m mae")
