import sys
import numpy as np
import verif.driver
import verif.bokeh_server


def main():
   verif.driver.run(sys.argv)


def xmain():
   verif.bokeh_server.main()

if __name__ == '__main__':
   main()
