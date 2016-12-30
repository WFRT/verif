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

class TestMae(unittest.TestCase):
   def _get(self):
      inputs = [verif.input.get_input(file) for file in ["examples/raw.txt", "examples/kf.txt"]]
      data = verif.data.Data(inputs=inputs)
      metric = verif.metric.Mae()
      output = verif.output.Standard(metric)
      return data,output

   def test_leadtime(self):
      data,output = self._get()
      x,y,_,_ = output._get_x_y(data, verif.axis.Leadtime())
      lt = data.leadtimes
      np.testing.assert_array_equal(x, lt)
      self.assertEqual(2, y.shape[1])
      self.assertEqual(len(lt), y.shape[0])

   def test_location(self):
      data,output = self._get()
      x,y,_,_ = output._get_x_y(data, verif.axis.Location())
      loc = data.locations
      self.assertEqual(len(loc), len(x))
      self.assertEqual(2, y.shape[1])
      self.assertEqual(len(loc), y.shape[0])

if __name__ == '__main__':
   unittest.main()
