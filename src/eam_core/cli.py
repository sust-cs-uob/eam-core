"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -meam_core` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``eam_core.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``eam_core.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import sys

import argparse

from eam_core.yaml_runner import setup_parser, run
import eam_core.log_configuration as logconf

logconf.config_logging()

import logging

logger = logging.getLogger()


def main(args=None):
    args = setup_parser(sys.argv[1:])
    logger.info(f"Running with parameters {args}")
    if args.verbose:
        level = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    runners = run(args)
