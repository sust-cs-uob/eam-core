import unittest
from os import path, chdir, getcwd


def get_static_path(filename):
    """
    The working directory changes depending on how tests are run.
    Since tests call models and xlsx files with static paths, this can lead to many tests incorrectly failing
    Instead, get the current script directory- which should point to /tests- and join it with the desired filename.

    :param filename: The file to be loaded by the test, with pathing relative to /tests
    :return: A more reliable static path to that file
    """
    directory = path.dirname(path.realpath(__file__))
    return path.join(directory, filename)


def set_cwd_to_script_dir():
    cwd = getcwd()
    directory = path.dirname(path.realpath(__file__))
    chdir(directory)
    return cwd


def return_to_base_cwd(cwd):
    chdir(cwd)


class MyTestCase(unittest.TestCase):

    def test_get_static_path(self):
        assert get_static_path('directory_test_controller.py') == __file__
        assert path.isfile(get_static_path('data/test_data.xlsx'))

