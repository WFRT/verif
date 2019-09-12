import unittest
import verif.metric_type
import numpy as np


class TestMetricType(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(verif.metric_type.Deterministic(), verif.metric_type.Deterministic())
        self.assertTrue(verif.metric_type.Deterministic() == verif.metric_type.Deterministic())
        self.assertFalse(verif.metric_type.Deterministic() != verif.metric_type.Deterministic())

    def test_unequal(self):
        self.assertFalse(verif.metric_type.Deterministic() == verif.metric_type.Probabilistic())
        self.assertFalse(verif.metric_type.Deterministic() == verif.metric_type.Probabilistic())
        self.assertTrue(verif.metric_type.Deterministic() != verif.metric_type.Probabilistic())
        self.assertTrue(verif.metric_type.Deterministic() != verif.metric_type.Probabilistic())


if __name__ == '__main__':
    unittest.main()
