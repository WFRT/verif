# Taken from https://gist.github.com/swenson/8142788
import os
import os.path
import unittest
import pep8
import warnings

# ignore stuff in virtualenvs or version control directories
ignore_patterns = ('.svn', 'bin', 'lib' + os.sep + 'python', 'verif/devel', 'tests/devel', 'devel')


def ignore(dir):
   """Should the directory be ignored?"""
   for pattern in ignore_patterns:
      if pattern in dir:
         return True
   return False


class TestPep8(unittest.TestCase):
   def test_pep8(self):
      style = pep8.StyleGuide(quiet=False)
      style.options.ignore += ('E111',)  # 4-spacing is just too much
      style.options.ignore += ('E501',)
      style.options.ignore += ('E114',)
      style.options.ignore += ('E121',)
      style.options.ignore += ('E122',)
      style.options.ignore += ('E126',)
      style.options.ignore += ('E127',)
      style.options.ignore += ('E128',)
      # style.options.max_line_length = 100  # because it isn't 1928 anymore

      errors = 0
      for dir in ['verif/', 'verif/tests/']:
         for root, _, files in os.walk(dir):
            if ignore(root):
               continue

            python_files = [os.path.join(root, f) for f in files if f.endswith('.py')]
            report = style.check_files(python_files)
            errors += report.total_errors

      if errors > 0:
         warnings.warn('Warning: There are %d PEP8 style errors in the source files' % errors)

if __name__ == "__main__":
   unittest.main()
