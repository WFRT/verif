#!/bin/env/python

# From: https://stackoverflow.com/a/33012308
# Runs coveralls if Travis CI is detected

import os
from subprocess import call

if __name__ == '__main__':
    if 'TRAVIS' in os.environ:
        rc = call('coveralls')
        raise SystemExit(rc)
    else:
        print("Travis was not detected -> Skipping coveralls")
