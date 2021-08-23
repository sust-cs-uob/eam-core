import unittest
#from tests.directory_test_controller import get_static_path, set_cwd_to_script_dir, return_to_base_cwd, use_test_dir

from math import isclose

from tests.directory_test_controller import get_static_path, use_test_dir
from eam_core.yaml_runner import setup_parser, run
from eam_core.util import load_as_df_quantity


def load_output_var(runners, variable):
    q_data, _ = load_as_df_quantity(f'{runners["default"].sim_control.output_directory}/result_data_{variable}.hdf5')
    df = q_data.pint.dequantify()
    df.columns = df.columns.droplevel(1)
    return df.mean(level='time').sum().to_dict()


class MyTestCase(unittest.TestCase):

    @unittest.skip("too much effort to maintain")
    def test_youtube(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a dev', '-d', get_static_path('models/youtube.yml')]))

    def test_ci_v2(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2.yml')]))
            output = load_output_var(runners, 'energy')

            print(output)

            assert isclose(output['CDN'], 6.500000000000002e-07)
            assert isclose(output['Internet Network'], 6.500000000000002e-07)
            assert isclose(output['Laptop'], 0.0026000000000000007)

    def test_ci_v2_linear(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2_linear.yml')]))
            output = load_output_var(runners, 'energy')

            print(output)

            assert isclose(output['CDN'], 0.000000845, rel_tol=0, abs_tol=0.000000001)
            assert isclose(output['Internet Network'], 0.000000845, rel_tol=0, abs_tol=0.000000001)
            assert isclose(output['Laptop'], 0.00338, rel_tol=0, abs_tol=0.00001)

    @unittest.skip("multithreading is unused, would only be for scenarios")
    def test_ci_v2_multithreaded(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-m', '-id', get_static_path('models/ci_v2.yml')]))

    def test_countries(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', get_static_path('models/ci_v2_group.yml')]))
            output = load_output_var(runners, 'energy')

            print(output)

            assert isclose(output['CDN'], 2.4375000000000005e-07)
            assert isclose(output['Internet Network'], 2.4375000000000005e-07)
            assert isclose(output['Laptop'], 0.0020222222222222226)


if __name__ == '__main__':
    unittest.main()
