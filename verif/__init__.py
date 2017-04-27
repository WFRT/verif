import sys
import numpy as np
import verif.driver
import verif.driver_bokeh


def main():
   verif.driver.run(sys.argv)


def xmain():
   verif.driver_bokeh.main()

if __name__ == '__main__':
   main()
