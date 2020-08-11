import unittest

from eam_core.yaml_runner import setup_parser, run


class MyTestCase(unittest.TestCase):
    def test_youtube(self):
        runners = run(setup_parser(['-l', '-a dev', '-d', 'tests/models/youtube.yml']))

    def test_ci_v2(self):
        runners = run(setup_parser(['-l', '-a', 'ci','-d','-id', 'tests/models/ci_v2.yml']))

    def test_countries(self):
        runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', 'tests/models/countriestest.yml']))

if __name__ == '__main__':
    unittest.main()
