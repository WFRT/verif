import unittest
import verif.aggregator
import numpy as np
import numbers


class TestAggregator(unittest.TestCase):
    def test_simple(self):
        agg = verif.aggregator.Mean()
        self.assertEqual(2, agg(np.array([1, 2, 3])))

    def test_name(self):
        mean = verif.aggregator.Mean()
        self.assertEqual("mean", mean.name())
        mean2 = verif.aggregator.get("mean")
        self.assertEqual("mean", mean2.name())
        self.assertEqual(mean, mean2)

    def test_equal(self):
        self.assertEqual(verif.aggregator.Mean(), verif.aggregator.Mean())
        self.assertTrue(verif.aggregator.Mean() == verif.aggregator.Mean())
        self.assertFalse(verif.aggregator.Mean() != verif.aggregator.Mean())

        self.assertEqual(verif.aggregator.Quantile(0.2), verif.aggregator.Quantile(0.2))
        self.assertTrue(verif.aggregator.Quantile(0.2) == verif.aggregator.Quantile(0.2))
        self.assertFalse(verif.aggregator.Quantile(0.2) != verif.aggregator.Quantile(0.2))

    def test_unequal(self):
        self.assertFalse(verif.aggregator.Mean() == verif.aggregator.Absmean())
        self.assertFalse(verif.aggregator.Mean() == verif.aggregator.Quantile(0.2))
        self.assertTrue(verif.aggregator.Mean() != verif.aggregator.Absmean())
        self.assertTrue(verif.aggregator.Mean() != verif.aggregator.Quantile(0.2))

        self.assertFalse(verif.aggregator.Quantile(0.2) == verif.aggregator.Quantile(0.3))
        self.assertTrue(verif.aggregator.Quantile(0.2) != verif.aggregator.Quantile(0.3))

    def test_call(self):
        aggregators = [verif.aggregator.Mean(), verif.aggregator.Min()]
        aggregators = list()
        for a in verif.aggregator.get_all():
            if a == verif.aggregator.Quantile:
                aggregators += [a(0.9)]
            else:
                aggregators += [a()]

        array = np.array([[0, 1, 2], [10, 11.0, 12]])
        for aggregator in aggregators:
            value = aggregator(array)
            self.assertTrue(isinstance(value, numbers.Number))
            value = aggregator(array, axis=0)
            self.assertEqual(value.shape, (3,))
            value = aggregator(array, axis=1)
            self.assertEqual(value.shape, (2,))


if __name__ == '__main__':
    unittest.main()
