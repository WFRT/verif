import unittest
import verif.data
import numpy as np


def get_data_from_text(filename):
    data = verif.data.Data(verif.input.Text(filename))
    return data


def assert_set_equal(array1, array2):
    np.testing.assert_array_equal(np.sort(array1), np.sort(array2))


class TestData(unittest.TestCase):
    def test_simple(self):
        inputs = [verif.input.get_input(filename) for filename in ["verif/tests/files/file2_ens.txt", "verif/tests/files/file3_ens.txt"]]
        data = verif.data.Data(inputs=inputs)

        fields = [verif.field.Ensemble()]
        axis = verif.axis.Leadtime()

        # 3 members
        ens, = data.get_scores(fields, 0, axis, 0)
        self.assertEqual((3,3), ens.shape)
        np.testing.assert_array_almost_equal([3,4,5], ens[0,:])

        # 2 members
        ens, = data.get_scores(fields, 1, axis, 0)
        self.assertEqual((3,2), ens.shape)
        np.testing.assert_array_almost_equal([1,2], ens[0,:])

    def test_preaggregated(self):
        inputs = [verif.input.get_input(filename) for filename in ["verif/tests/files/file2_ens.txt", "verif/tests/files/file3_ens.txt"]]
        data = verif.data.Data(inputs=inputs, dim_agg_length=12, dim_agg_method=verif.aggregator.Sum())

        fields = [verif.field.Ensemble()]
        axis = verif.axis.Leadtime()

        # Look for the sum of leadtime 0 and 1
        leadtime = 1

        # 3 members
        ens, = data.get_scores(fields, 0, axis, leadtime)
        self.assertEqual((3,3), ens.shape)
        np.testing.assert_array_almost_equal([7,8,9], ens[0,:])

        # 2 members
        ens, = data.get_scores(fields, 1, axis, leadtime)
        self.assertEqual((3,2), ens.shape)
        np.testing.assert_array_almost_equal([4,6], ens[0,:])



if __name__ == '__main__':
    unittest.main()
