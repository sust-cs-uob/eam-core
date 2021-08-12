import unittest
#from tests.directory_test_controller import get_static_path, set_cwd_to_script_dir, return_to_base_cwd, use_test_dir
from tests.directory_test_controller import get_static_path, use_test_dir
from eam_core.yaml_runner import setup_parser, run


class MyTestCase(unittest.TestCase):

    @unittest.skip("too much effort to maintain")
    def test_youtube(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a dev', '-d', get_static_path('models/youtube.yml')]))

    def test_ci_v2(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2.yml')]))

    @unittest.skip("multithreading is unused, would only be for scenarios")
    def test_ci_v2_multithreaded(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-m', '-id', get_static_path('models/ci_v2.yml')]))

    def test_countries(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', get_static_path('models/countriestest.yml')]))


if __name__ == '__main__':
    unittest.main()
