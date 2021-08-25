import unittest
#from tests.directory_test_controller import get_static_path, set_cwd_to_script_dir, return_to_base_cwd, use_test_dir

from math import isclose

import numpy as np

from tests.directory_test_controller import get_static_path, use_test_dir
from eam_core.yaml_runner import setup_parser, run
from eam_core.util import load_as_df_quantity

import pandas as pd
from pandas.testing import assert_frame_equal

import pint
import pint_pandas

from datetime import datetime


def load_output_dataframe(runners, variable):
    q_data, _ = load_as_df_quantity(f'{runners["default"].sim_control.output_directory}/result_data_{variable}.hdf5')
    return q_data


def get_means(output):
    df = output.pint.dequantify()
    df.columns = df.columns.droplevel(1)
    return df.mean(level='time').sum().to_dict()


def assert_pint_frame_equal(a, b, **kwargs):
    assert_frame_equal(a.pint.dequantify(), b.pint.dequantify(), **kwargs)


dates = [
    datetime(2019, 1, 1),
    datetime(2019, 2, 1),
    datetime(2019, 3, 1),
    datetime(2019, 4, 1),
    datetime(2019, 5, 1),
    datetime(2019, 6, 1),
    datetime(2019, 7, 1),
    datetime(2019, 8, 1),
    datetime(2019, 9, 1),
    datetime(2019, 10, 1),
    datetime(2019, 11, 1),
    datetime(2019, 12, 1),
    datetime(2020, 1, 1)
]


class TestModels(unittest.TestCase):

    @unittest.skip("too much effort to maintain")
    def test_youtube(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a dev', '-d', get_static_path('models/youtube.yml')]))

    def test_ci_v2(self):
        expected_index = pd.MultiIndex.from_arrays([dates, np.zeros(13, dtype=int)], names=['time', 'samples'])

        expected = pd.DataFrame(index=expected_index)

        expected['CDN'] = pd.Series(index=expected_index, data=np.full(13, 5.000000000000002e-08),
                                    dtype='pint[gigawatt_hour]')
        expected['Internet Network'] = pd.Series(index=expected_index, data=np.full(13, 5.000000000000002e-08),
                                                 dtype='pint[gigawatt_hour]')
        expected['Laptop'] = pd.Series(index=expected_index, data=np.full(13, 0.00020000000000000006),
                                       dtype='pint[gigawatt_hour]')

        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2.yml')]))
            output = load_output_dataframe(runners, 'energy')

            assert_pint_frame_equal(output, expected)

            means = get_means(output)

            print(means)

            assert isclose(means['CDN'], 6.500000000000002e-07)
            assert isclose(means['Internet Network'], 6.500000000000002e-07)
            assert isclose(means['Laptop'], 0.0026000000000000007)

    def test_ci_v2_linear(self):
        expected_index = pd.MultiIndex.from_arrays([dates, np.zeros(13, dtype=int)], names=['time', 'samples'])

        expected = pd.DataFrame(index=expected_index)

        expected['CDN'] = pd.Series(index=expected_index, data=[
            5.000E-08,
            5.250E-08,
            5.500E-08,
            5.750E-08,
            6.000E-08,
            6.250E-08,
            6.500E-08,
            6.750E-08,
            7.000E-08,
            7.250E-08,
            7.500E-08,
            7.750E-08,
            8.000E-08
        ], dtype='pint[gigawatt_hour]')

        expected['Internet Network'] = pd.Series(index=expected_index, data=[
            5.000E-08,
            5.250E-08,
            5.500E-08,
            5.750E-08,
            6.000E-08,
            6.250E-08,
            6.500E-08,
            6.750E-08,
            7.000E-08,
            7.250E-08,
            7.500E-08,
            7.750E-08,
            8.000E-08
        ], dtype='pint[gigawatt_hour]')

        expected['Laptop'] = pd.Series(index=expected_index, data=[
            0.000200,
            0.000210,
            0.000220,
            0.000230,
            0.000240,
            0.000250,
            0.000260,
            0.000270,
            0.000280,
            0.000290,
            0.000300,
            0.000310,
            0.000320
        ], dtype='pint[gigawatt_hour]')

        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2_linear.yml')]))
            output = load_output_dataframe(runners, 'energy')

            assert_pint_frame_equal(output, expected, rtol=1e-2)

            means = get_means(output)

            print(means)

            assert isclose(means['CDN'], 0.000000845, rel_tol=0, abs_tol=0.000000001)
            assert isclose(means['Internet Network'], 0.000000845, rel_tol=0, abs_tol=0.000000001)
            assert isclose(means['Laptop'], 0.00338, rel_tol=0, abs_tol=0.00001)

    def test_ci_v2_exp(self):
        expected_index = pd.MultiIndex.from_arrays([dates, np.zeros(13, dtype=int)], names=['time', 'samples'])

        expected = pd.DataFrame(index=expected_index)

        expected['CDN'] = pd.Series(index=expected_index, data=[
            5.00000000000000E-08,
            5.29731547179648E-08,
            5.61231024154686E-08,
            5.94603557501360E-08,
            6.29960524947437E-08,
            6.67419927085017E-08,
            7.07106781186548E-08,
            7.49153538438341E-08,
            7.93700525984100E-08,
            8.40896415253715E-08,
            8.90898718140339E-08,
            9.43874312681693E-08,
            1.00000000000000E-07
        ], dtype='pint[gigawatt_hour]')

        expected['Internet Network'] = pd.Series(index=expected_index, data=[
            5.00000000000000E-08,
            5.29731547179648E-08,
            5.61231024154686E-08,
            5.94603557501360E-08,
            6.29960524947437E-08,
            6.67419927085017E-08,
            7.07106781186548E-08,
            7.49153538438341E-08,
            7.93700525984100E-08,
            8.40896415253715E-08,
            8.90898718140339E-08,
            9.43874312681693E-08,
            1.00000000000000E-07
        ], dtype='pint[gigawatt_hour]')

        expected['Laptop'] = pd.Series(index=expected_index, data=[
            0.000200000000000000,
            0.000211892618871859,
            0.000224492409661875,
            0.000237841423000544,
            0.000251984209978975,
            0.000266967970834007,
            0.000282842712474619,
            0.000299661415375336,
            0.000317480210393640,
            0.000336358566101486,
            0.000356359487256136,
            0.000377549725072677,
            0.000400000000000000
        ], dtype='pint[gigawatt_hour]')

        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-id', get_static_path('models/ci_v2_exp.yml')]))
            output = load_output_dataframe(runners, 'energy')

            assert_pint_frame_equal(output, expected)#, rtol=1e-2)

            means = get_means(output)

            print(means)

            assert isclose(means['CDN'], 0.000000940857687255288)#, rel_tol=0, abs_tol=0.000000001)
            assert isclose(means['Internet Network'], 0.000000940857687255288)#, rel_tol=0, abs_tol=0.000000001)
            assert isclose(means['Laptop'], 0.00376343074902115)#, rel_tol=0, abs_tol=0.00001)

    @unittest.skip("multithreading is unused, would only be for scenarios")
    def test_ci_v2_multithreaded(self):
        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', '-m', '-id', get_static_path('models/ci_v2.yml')]))

    def test_countries(self):
        expected_index = pd.MultiIndex.from_product([dates, [0], ['A', 'B']], names=['time', 'samples', 'group'])

        expected = pd.DataFrame(index=expected_index)

        expected['CDN'] = pd.Series(index=expected_index, data=np.tile([2.5E-08, 1.25E-08], 13),
                                    dtype='pint[gigawatt_hour]')
        expected['Internet Network'] = pd.Series(index=expected_index, data=np.tile([2.5E-08, 1.25E-08], 13),
                                                 dtype='pint[gigawatt_hour]')
        expected['Laptop'] = pd.Series(index=expected_index, data=np.tile([2E-04, 1.11111111111E-04], 13),
                                       dtype='pint[gigawatt_hour]')

        with use_test_dir():
            runners = run(setup_parser(['-l', '-a', 'ci', '-d', get_static_path('models/ci_v2_group.yml')]))
            output = load_output_dataframe(runners, 'energy')

            assert_pint_frame_equal(output, expected)

            means = get_means(output)

            print(means)

            assert isclose(means['CDN'], 2.4375000000000005e-07)
            assert isclose(means['Internet Network'], 2.4375000000000005e-07)
            assert isclose(means['Laptop'], 0.0020222222222222226)


if __name__ == '__main__':
    unittest.main()
