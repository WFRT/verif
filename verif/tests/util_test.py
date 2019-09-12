from __future__ import print_function
import six

import datetime
import matplotlib.dates
import numpy as np
import unittest
import verif.util


class TestApplyThreshold(unittest.TestCase):
    def test_1(self):
        ar = np.array([1, 3, 2])
        np.testing.assert_array_equal(np.array([0, 1, 0]), verif.util.apply_threshold(ar, "above", 2))
        np.testing.assert_array_equal(np.array([0, 1, 1]), verif.util.apply_threshold(ar, "above=", 2))
        np.testing.assert_array_equal(np.array([1, 0, 0]), verif.util.apply_threshold(ar, "below", 2))
        np.testing.assert_array_equal(np.array([1, 0, 1]), verif.util.apply_threshold(ar, "below=", 2))


class TestParseNumbers(unittest.TestCase):
    def test_simple(self):
        self.assertEqual([2], verif.util.parse_numbers("2"))
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("test")

    def test_vector(self):
        self.assertEqual([2, 3, 4, 5], verif.util.parse_numbers("2:5"))
        self.assertEqual([], verif.util.parse_numbers("2:1"))
        with self.assertRaises(SystemExit):
            self.assertEqual([2], verif.util.parse_numbers("2:test"))

    def test_vectorInc(self):
        self.assertEqual([2, 5, 8], verif.util.parse_numbers("2:3:8"))
        self.assertEqual([2, 5], verif.util.parse_numbers("2:3:7"))
        self.assertEqual([], verif.util.parse_numbers("2:-1:7"))
        self.assertEqual([2, 1, 0], verif.util.parse_numbers("2:-1:0"))
        self.assertEqual([8, 5, 2], verif.util.parse_numbers("8:-3:0"))
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2:3:test")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("test:3:5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2:test:5")

    def test_comma(self):
        self.assertEqual([2, 5], verif.util.parse_numbers("2,5"))
        self.assertEqual([3, 3], verif.util.parse_numbers("3,3"))
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("test")

    def test_decimal(self):
        self.assertEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], verif.util.parse_numbers("0.1:0.1:0.7"))
        self.assertEqual([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4], verif.util.parse_numbers("0.0001:0.0001:0.0007"))
        self.assertEqual([1e-6, 3e-6], verif.util.parse_numbers("0.000001,0.000003"))

    def test_0_step(self):
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("0:0:5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("0:0:0")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("0:0:-5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("20150101:0:20150101", is_date=True)
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("20150101:0:20150105", is_date=True)
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("20150101:0:20140101", is_date=True)

    def test_mix(self):
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2,5:8,3,test")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2,5:8,test,5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2,5:test,3,5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2,5:test,3,5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("2,test:8,3,5")
        with self.assertRaises(SystemExit):
            verif.util.parse_numbers("test,5:8,3,5")

    def test_date(self):
        self.assertEqual([20141230, 20141231, 20150101, 20150102, 20150103], verif.util.parse_numbers("20141230:20150103", True))
        self.assertEqual([20141230, 20150101, 20150103], verif.util.parse_numbers("20141230:2:20150104", True))

    def test_unixtime_to_datenum(self):
        unixtimes = np.array([1331856000, -2180131200])
        datenums = [verif.util.unixtime_to_datenum(unixtime) for unixtime in unixtimes]
        self.assertEqual(matplotlib.dates.date2num(datetime.datetime(2012, 3, 16, 0, 0)), datenums[0])
        self.assertEqual(matplotlib.dates.date2num(datetime.datetime(1900, 12, 1, 0, 0)), datenums[1])

    def test_date_to_unixtime(self):
        import time
        s = time.time()
        self.assertEqual(1475280000, verif.util.date_to_unixtime_slow(20161001))
        e = time.time()
        print (e - s)
        s = time.time()
        self.assertEqual(1475280000, verif.util.date_to_unixtime(20161001))
        e = time.time()
        print (e - s)

    def test_unixtime_to_date(self):
        self.assertEqual(20161001, verif.util.unixtime_to_date(1475280000))


class TestGetDate(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(20150207, verif.util.get_date(20150206, 1))
        self.assertEqual(20150205, verif.util.get_date(20150206, -1))
        self.assertEqual(20150215, verif.util.get_date(20150210, 5))

    def test_endofyear(self):
        self.assertEqual(20150101, verif.util.get_date(20141231, 1))
        self.assertEqual(20141226, verif.util.get_date(20150105, -10))
        self.assertEqual(20150105, verif.util.get_date(20141226, 10))
        self.assertEqual(20141231, verif.util.get_date(20150101, -1))


class TestGetIntervals(unittest.TestCase):
    def test_above(self):
        thresholds = [0, 1, 5]
        intervals = verif.util.get_intervals("above", thresholds)
        self.assertEqual(verif.interval.Interval(0, np.inf, False, False), intervals[0])

    def test_within(self):
        thresholds = [0, 1, 5]
        intervals = verif.util.get_intervals("within", thresholds)
        self.assertEqual(2, len(intervals))
        self.assertEqual(verif.interval.Interval(0, 1, False, False), intervals[0])
        self.assertEqual(verif.interval.Interval(1, 5, False, False), intervals[1])

    def test_within_single(self):
        thresholds = [0]
        intervals = verif.util.get_intervals("within", thresholds)
        self.assertEqual(0, len(intervals))


class TestAlmostEqual(unittest.TestCase):
    def test_1(self):
        for tol in [10**(-i) for i in range(1, 10)]:
            self.assertTrue(verif.util.almost_equal(0, 0, tol))
            self.assertTrue(verif.util.almost_equal(3, 3, tol))
            self.assertTrue(verif.util.almost_equal(-2.3, -2.3, tol))
            self.assertFalse(verif.util.almost_equal(-2.3, 2.3, tol))
            self.assertFalse(verif.util.almost_equal(-2.3, 0, tol))
            self.assertFalse(verif.util.almost_equal(0, 15.2, tol))

        self.assertTrue(verif.util.almost_equal(3.12345678, 3.12345679, 1e-1))
        self.assertTrue(verif.util.almost_equal(3.12345678, 3.12345679, 1e-5))
        self.assertTrue(verif.util.almost_equal(3.12345678, 3.12345679, 1e-8))
        self.assertFalse(verif.util.almost_equal(3.12345678, 3.12345679, 1e-9))
        self.assertTrue(verif.util.almost_equal(3.123, 3.1241, 1e-1))
        self.assertTrue(verif.util.almost_equal(3.123, 3.1241, 1e-2))
        self.assertFalse(verif.util.almost_equal(3.123, 3.1241, 1e-3))


class TestConvert(unittest.TestCase):
    def test_convert_back_and_forth(self):
        dates = [20150101, 20141231]
        new_dates = [verif.util.datenum_to_date(verif.util.date_to_datenum(date)) for date in dates]
        six.assertCountEqual(self, new_dates, dates)


class TestIsValidNc(unittest.TestCase):
    def test_1(self):
        self.assertTrue(verif.util.is_valid_nc("verif/tests/files/netcdf_valid1.nc"))
        self.assertFalse(verif.util.is_valid_nc("verif/tests/files/file1.txt"))


class TestProj4(unittest.TestCase):
    def test_1(self):
        string = "+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +no_defs +R=6.371e+07"
        r = verif.util.proj4_string_to_dict(string)
        self.assertEqual(r.get("+proj"), "lcc")
        self.assertEqual(r.get("+lat_0"), 63)
        self.assertEqual(r.get("+no_defs"), True)
        self.assertEqual(r.get("+lon_1"), None)


class TestDistance(unittest.TestCase):
    def test_1(self):
        # Verified against http://www.nhc.noaa.gov/gccalc.shtml
        # They only give answer to nearest km, so allow some deviation
        self.assertEqual(0, verif.util.distance(60, 10, 60, 10))
        self.assertLess(abs(1360000 - verif.util.distance(50.5, 3.4, 61.9, 11.5)), 2000)
        self.assertLess(abs(15712000 - verif.util.distance(-47.2, -24.4, 82.1, 101.5)), 2000)


if __name__ == '__main__':
    unittest.main()
