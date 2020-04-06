========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/eam-core/badge/?style=flat
    :target: https://readthedocs.org/projects/eam-core
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/dschien/eam-core.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dschien/eam-core

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/dschien/eam-core?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/dschien/eam-core

.. |requires| image:: https://requires.io/github/dschien/eam-core/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/dschien/eam-core/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/dschien/eam-core/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/dschien/eam-core

.. |version| image:: https://img.shields.io/pypi/v/eam-core.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/eam-core

.. |wheel| image:: https://img.shields.io/pypi/wheel/eam-core.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/eam-core

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/eam-core.svg
    :alt: Supported versions
    :target: https://pypi.org/project/eam-core

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/eam-core.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/eam-core

.. |commits-since| image:: https://img.shields.io/github/commits-since/dschien/eam-core/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/dschien/eam-core/compare/v0.0.0...master



.. end-badges

EAM core framework

Installation
============

::

    pip install eam-core

You can also install the in-development version with::

    pip install https://github.com/dschien/eam-core/archive/master.zip


Documentation
=============


https://eam-core.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
