import unittest
import numpy as np
import verif
import verif.input
import verif.data
import verif.location


class InputTextTest(unittest.TestCase):
    def test_ensemble_mean(self):
        input = verif.input.Text("verif/tests/files/file1_ens.txt")
        print(input.ensemble[0, 0, 0, :])
        mean = input.ensemble_mean
        self.assertEqual((3, 3, 2, 3), input.ensemble.shape)
        self.assertEqual((3, 3, 2), mean.shape)

        np.testing.assert_array_equal((4, 4, 5), input.ensemble[0, 0, 0, :])
        self.assertAlmostEqual(4.333333333, mean[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
