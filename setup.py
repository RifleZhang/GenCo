
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup

def do_setup():
    setup(
        name="genco",
        version="0.0.1",
        packages=find_packages(
            include=[
               "xmclib",
            ]
        )
    )

do_setup()