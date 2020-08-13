========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |circleci|

.. |circleci| image:: https://circleci.com/gh/sust-cs-uob/eam-core.svg?style=svg&circle-token=952c0d872cff9a2534b23c9e25d269c033ec725d
    :alt: circleci Build Status
    :target: https://circleci.com/gh/sust-cs-uob/eam-core

.. end-badges

EAM core framework


Development
===========

Requires pandoc https://pandoc.org/installing.html

create new virtual env (e.g on the CLI)::

    python3 -m venv eam-core
    source eam-core/bin/activate

then install eam-core::

    pip install -e .


install non-pypi dependencies::

    pip install git+https://github.com/hgrecco/pint.git@f356379c15c1cb5d211c795872ac9e9284d2358f#egg=pint
    pip install --no-deps git+https://github.com/hgrecco/pint-pandas.git#egg=pint-pandas



To run the all tests run::

    tox


