import unittest
import verif.driver
import os
import numpy as np
import tempfile
np.seterr('raise')


all_axes = ["time", "leadtime", "timeofday", "dayofyear", "monthofyear", "day", "week", "month", "year", "leadtimeday", "location", "lat", "lon", "elev"]


class IntegrationTest(unittest.TestCase):
    """
    These tests run verif on the command-line, but do not test the validity of the
    output graphics, only that they do or do not create errors.
    """

    @staticmethod
    def run_command(command):
        """ Runs a verif command line """
        argv = command.split()
        verif.driver.run(argv)

    @staticmethod
    def remove(file):
        """ Removes a file """
        os.remove(file)

    @staticmethod
    def file_size(filename):
        """ Returns the number of bytes of a file """
        statinfo = os.stat(filename)
        return statinfo.st_size

    @staticmethod
    def is_valid_file(filename, min_size=3000):
        """ Checks if a file is larger in size than min_size bytes """
        return IntegrationTest.file_size(filename) > min_size

    def run_with_image(self, command):
        """
        Runs the verif command and appends -f <somefile>.png so that it will write output
        to a temporary png file. Removes the file afterwards.
        """
        fd, imageFile = tempfile.mkstemp(suffix=".png")
        command = command + " -f " + imageFile
        self.run_command(command)
        self.assertTrue(self.is_valid_file(imageFile), 3000)
        os.close(fd)
        self.remove(imageFile)

    def run_with_text(self, command):
        """
        Runs the verif command and appends -f <somefile>.txt so that it will write output
        to a temporary txt file. Removes the file afterwards.
        """
        fd, textFile = tempfile.mkstemp(suffix=".txt")
        command = command + " -f " + textFile
        self.run_command(command)
        self.assertTrue(self.is_valid_file(textFile, 10))
        os.close(fd)
        self.remove(textFile)

    def test_valid(self):
        self.run_command("verif")
        self.run_command("verif --version")
        self.run_command("verif examples/raw.txt examples/kf.txt --list-thresholds")
        self.run_command("verif examples/raw.txt examples/kf.txt --list-quantiles")
        self.run_command("verif examples/raw.txt examples/kf.txt --list-times")
        self.run_command("verif examples/raw.txt examples/kf.txt --list-dates")
        self.run_command("verif examples/raw.txt examples/kf.txt --list-thresholds --list-quantiles --list-times")

    def test_invalid(self):
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-thresholds")
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-quantiles")
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-times")

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
        self.run_with_image("verif -leg 1,2 examples/raw.txt examples/kf.txt -m ets -x no")
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

    def test_standard_option_x(self):
        for axis in all_axes:
            self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x %s" % axis)

    def test_obsfcst_option_x(self):
        for axis in all_axes:
            self.run_with_image("verif examples/raw.txt examples/kf.txt -m obsfcst -x %s" % axis)

    def test_pithist(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m pithistdev")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m pithistslope")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m pithistshape")

    def test_obs_subset(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -r 10")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -x threshold")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -r 0,2,10 -x threshold -b within")

    def test_annotate(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -a")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -a -x location")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -a -type map")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m obsfcst -a")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m obsfcst -a -x location")

    def test_plotting_options(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -aspect 0.1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -aspect 2.1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -bottom 0.1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -bottom 0.5")
        # -clim and -cmap are tested with -type map
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -dpi 50")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -dpi 300")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -fs 10,2")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -labfs 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -labfs 11")
        # -lc tests are in separate functions
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -left 0.8")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -legfs 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -legfs 10")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -legloc right")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -legloc lower_left")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ls - -ma ,o")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ls None -ma *")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ls -,-, -ma ,s,:")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lw 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lw 1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lw 1.3")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lw 2")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ms 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ms 1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ms 1.3")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ms 2")
        # For some reason this fails without -left 0.1, although it works fine when verif is
        # invoked on the command line:
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -left 0.1 -right 0.8")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -tickfs 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -tickfs 10")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -title title")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -titlefs 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -titlefs 10")
        # Same as for -right above
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -bottom 0.1 -top 0.4")
        # -type is tested separately
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xlabel test")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xlim 0,1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xrot 90")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xticks 0:4")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xticks 0:3 -xticklabels 0,test,1,2")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xticklabels 0,test,1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -xticklabels ''")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ylabel test")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -ylim 0,1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -yrot 90")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -yticks 0:4")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -yticks 0:3 -yticklabels 0,test,1,2")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -yticklabels 0,test,1")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -yticklabels ''")

    def test_against(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m against")
        # Ensure at least 3 files to test the subplots
        self.run_with_image("verif examples/raw.txt examples/kf.txt examples/raw.txt -m against")

    def test_impact(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type impact")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m corr -type impact -ms 6 -r 0:0.1:1")

    def test_mapimpact(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type mapimpact -legfs 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m ets -type mapimpact -r 1 -legfs 0")

    def test_fss(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m fss -r 5")
        self.run_with_image("verif verif/tests/files/file3.txt -m fss -r 5")
        self.run_with_image("verif verif/tests/files/file3.txt -m fss -r 100")
        self.run_with_image("verif verif/tests/files/file3.txt -m fss -r 0.1 -x leadtime")
        with self.assertRaises(SystemExit):
            self.run_with_image("verif verif/tests/files/file3.txt -m fss -r 0.1 -x time")

    def test_taylor(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m taylor -xlim 0,2")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m taylor -xlim 0,0.2")

    def test_roc(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m droc -r 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m droc -r 0 -simple")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m droc -r 0 -xlog")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m droc -r 0 -ylog")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m droc -r 0 -xlog -ylog")

    def test_obsleg(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m obsfcst -obsleg Test -leg 1,2")

    def test_discrimination(self):
        self.run_with_image("verif examples/raw.txt -m discrimination -r 0")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m discrimination -r 0")

    def test_scatter(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m scatter")

    def test_auto(self):
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m autocorr")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m autocov")

    def test_autocorr(self):
        self.run_with_image("verif examples/raw.txt -m autocorr")
        self.run_with_image("verif examples/raw.txt -m autocorr -x location")
        self.run_with_image("verif examples/raw.txt -m autocorr -x lat")
        self.run_with_image("verif examples/raw.txt -m autocorr -x lon")
        self.run_with_image("verif examples/raw.txt -m autocorr -x elev")
        self.run_with_image("verif examples/raw.txt -m autocorr -x time")
        self.run_with_image("verif examples/raw.txt -m autocorr -x leadtime")
        self.run_with_image("verif examples/raw.txt -m autocorr -r 0:100:1000")
        self.run_with_image("verif examples/raw.txt -m autocorr -r 0:100:1000 -xlim 0,100")

    def test_config(self):
        self.run_with_image("verif examples/raw.txt --config verif/tests/files/config1.txt")
        self.run_with_image("verif examples/raw.txt -m mae --config verif/tests/files/configEmpty.txt")

    def test_other_fields(self):
        self.run_with_image("verif verif/tests/files/file1_crps.txt -m crps")
        self.run_with_image("verif verif/tests/files/file1_crps.txt -m crps -x time")
        self.run_with_image("verif verif/tests/files/file1_crps.txt -m crps -agg median")

    def test_map_type(self):
        pass

    def test_type(self):
        self.run_with_text("verif examples/raw.txt examples/kf.txt -m mae -type text")
        self.run_with_text("verif examples/raw.txt examples/kf.txt -m mae -type csv")
        # These cause a FutureWarning in mpl, but not much we can do about that
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type map -clim 0,11")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type map -cmap RdBu")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type map")
        self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -type maprank")

    def test_freq(self):
        self.run_with_image("verif verif/tests/files/file1.txt -m freq")
        # Check that no error occurs, even though fcst or obs is not available
        self.run_with_image("verif verif/tests/files/file1_no_obs.txt -m freq")
        self.run_with_image("verif verif/tests/files/file1_no_fcst.txt -m freq")

    def test_option_lc(self):
        for lc in ("g,r", "g", "g,r,b", "0,0.5,0.9", "[0,0,1],0.5,g"):
            self.run_with_image("verif examples/raw.txt examples/kf.txt -m mae -lc %s" % lc)

    def test_boolean_options(self):
        for opt in ("acc", "nogrid", "nomargin", "hist", "sort", "sp", "simple", "xlog", "ylog"):
            self.run_with_image("verif examples/raw.txt examples/kf.txt -m obs -%s" % opt)

    def test_invalidMetric(self):
        with self.assertRaises(SystemExit):
            self.run_with_image("verif examples/T_raw_0.nc -m maeq")

    def test_invalidFile(self):
        with self.assertRaises(SystemExit):
            self.run_with_image("verif examples/T_raw_1.nc -m mae")


if __name__ == '__main__':
    unittest.main()
