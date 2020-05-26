import unittest

from eam_core.yaml_runner import setup_parser, run


class MyTestCase(unittest.TestCase):
    def test_something(self):
        runners = run(setup_parser(['-l', '--skip_storage', '-v', 'models/structure_test.yml']))


if __name__ == '__main__':
    unittest.main()
