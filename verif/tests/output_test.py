import unittest
import verif.data
import verif.output
import verif.metric
import numpy as np


class TestOutput(unittest.TestCase):
   def test_text(self):
     inputs = [verif.input.get_input("verif/tests/files/file_sigdig.txt")]
     data = verif.data.Data(inputs=inputs)
     metric = verif.metric.Obs()

     output = verif.output.Standard(metric)
     output.text(data)
