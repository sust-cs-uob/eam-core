import unittest
#from tests.directory_test_controller import get_static_path, set_cwd_to_script_dir, return_to_base_cwd, use_test_dir
from tests.directory_test_controller import get_static_path, use_test_dir
from eam_core.yaml_runner import setup_parser, run


class MyTestCase(unittest.TestCase):

    @unittest.skip("too much effort to maintain")
    def test_youtube(self):
        #cwd = set_cwd_to_script_dir()
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a dev', '-d', get_static_path('models/youtube.yml')]))
        #return_to_base_cwd(cwd)

    def test_ci_v2(self):
        #cwd = set_cwd_to_script_dir()
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2.yml')]))
        #return_to_base_cwd(cwd)

    def test_countries(self):
        #cwd = set_cwd_to_script_dir()
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', get_static_path('models/countriestest.yml')]))
        #return_to_base_cwd(cwd)


if __name__ == '__main__':
    unittest.main()
