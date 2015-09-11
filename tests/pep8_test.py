# Taken from https://gist.github.com/swenson/8142788
import os
import os.path
import unittest
import pep8

# ignore stuff in virtualenvs or version control directories
ignore_patterns = ('.svn', 'bin', 'lib' + os.sep + 'python', 'devel')


def ignore(dir):
  """Should the directory be ignored?"""
  for pattern in ignore_patterns:
    if pattern in dir:
      return True
  return False


class TestPep8(unittest.TestCase):
  def test_pep8(self):
    style = pep8.StyleGuide(quiet=True)
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
    for root, _, files in os.walk('verif/'):
      if ignore(root):
        continue

      python_files = [os.path.join(root, f) for f in files if f.endswith('.py')]
      errors += style.check_files(python_files).total_errors

    self.assertEqual(errors, 0, 'PEP8 style errors: %d' % errors)

