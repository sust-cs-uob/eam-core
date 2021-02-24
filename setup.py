#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='eam-core',
    version='0.1.0',
    description='EAM core framework',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Dan Schien',
    author_email='daniel.schien@bristol.ac.uk',
    url='https://github.com/sust-cs-uob/eam-core',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    # package_data=[('eam_core', [os.path.join('eam_core', 'logconf.yml')])],
    # data_files=[('', [os.path.join('eam_core', 'logconf.yml')])],
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
    ],
    project_urls={
        'Documentation': 'https://eam-core.readthedocs.io/',
        'Changelog': 'https://eam-core.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/dschien/eam-core/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.6',
    # install_requires=[
    #     # eg: 'aspectlib==1.1.1', 'six>=1.7',
    # ],
    # extras_require={
    #     'gdrive' = ['httplib2>=0.10.3', 'google-api-python-client', 'oauth2client'],
    # },
    entry_points={
        'console_scripts': [
            'eam-core = eam_core.cli:main',
        ]
    },
)
