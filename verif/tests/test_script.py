import unittest
import verif.driver
import os
import sys
import numpy as np
import tempfile
np.seterr('raise')


class ScriptTest(unittest.TestCase):
    """
    These tests run verif on the command-line, but do not test the validity of the
    output graphics, only that they do or do not create errors.
    """

    @staticmethod
    def run_python(command):
        """ Runs a verif command line """
        command = "%s %s" % (sys.executable, command)
        status = os.system(command)
        if status != 0:
            raise Exception("Command failed: %s" % command)
        return

    def test_valid(self):
        programs = ["text2nc", "accumulate", "ens2prob", "expandverif", "window"]
        for program in programs:
            self.run_python("scripts/%s.py" % program)

    def test_invalid(self):
        """
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-thresholds")
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-quantiles")
        with self.assertRaises(SystemExit):
            self.run_with_image("verif --list-times")
        """


if __name__ == '__main__':
    unittest.main()
