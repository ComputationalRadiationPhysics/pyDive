from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

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
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='pyDive',
    version=pyDive.__version__,
    url='http://github.com/ComputationalRadiationPhysics/pyDive',
    license='GNU Affero General Public License v3',
    author='Heiko Burau',
    #tests_require=['pytest'],
    install_requires=requirements,
    #cmdclass={'test': PyTest},
    author_email='h.burau@hzdr.de',
    description='Distributed Interactive Visualization and Exploration of large datasets',
    long_description=long_description,
    packages=['pyDive'],
    include_package_data=True,
    platforms='any',
    #test_suite='sandman.test.test_sandman',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha Development Status',
        'Natural Language :: English',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    #extras_require={
    #    'testing': ['pytest'],
    }
)