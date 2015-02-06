"""
Copyright 2014 Heiko Burau

This file is part of pyDive.

pyDive is free software: you can redistribute it and/or modify
it under the terms of of either the GNU General Public License or
the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
pyDive is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License and the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU General Public License
and the GNU Lesser General Public License along with pyDive.
If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys
import subprocess
import time

import pyDive

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')#, 'CHANGES.txt')
requirements = [line.rstrip('\n') for line in open('requirements.txt')]

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        profile_name = raw_input("Name of your IPython-parallel profile you want to run the tests with: ")
        n_engines = raw_input("Number of engines: ")

        # start ipcluster
        print("Waiting for engines to start...")
        subprocess.Popen(("ipcluster", "start", "--n=%s" % n_engines,\
            "--profile=%s" % profile_name))
        time.sleep(35)

        import pytest
        # set profile name as environment variable
        os.environ["IPP_PROFILE_NAME"] = profile_name
        errcode = pytest.main(self.test_args)

        # stop ipcluster
        subprocess.Popen(("ipcluster", "stop", "--profile=%s" % profile_name)).wait()

        sys.exit(errcode)

setup(
    name='pyDive',
    version=pyDive.__version__,
    url='http://github.com/ComputationalRadiationPhysics/pyDive',
    license='GNU Affero General Public License v3',
    author='Heiko Burau',
    tests_require=['pytest'],
    install_requires=requirements,
    cmdclass={'test': PyTest},
    author_email='h.burau@hzdr.de',
    description='Distributed Interactive Visualization and Exploration of large datasets',
    long_description=long_description,
    packages=['pyDive', 'pyDive/distribution', 'pyDive/arrays', 'pyDive/cloned_ndarray', 'pyDive/test'],
    include_package_data=True,
    platforms='any',
    #test_suite='sandman.test.test_sandman',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    extras_require={'testing': ['pytest'],}
)