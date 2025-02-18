from __future__ import print_function
import sys

try:
    from setuptools import setup
except ImportError:
    sys.stderr.write(
        "Please install setuptools before running this script. Exiting."
    )
    sys.exit(1)


if __name__ == '__main__':
    setup()
