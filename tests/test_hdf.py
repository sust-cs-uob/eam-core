import os

import numpy as np
import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from eam_core import Q_, ureg
from eam_core.util import h5store, h5load, load_as_df_qantity


class MyTestCase(unittest.TestCase):
    def test_basic(self):
        samples = 2
        date_range = pd.date_range('2016-01-01', '2016-12-01', freq='MS')

        df = pd.DataFrame(data=np.arange(len(date_range) * samples, dtype=float).ravel(), columns=['test_col'])
        iterables = [date_range, range(samples)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)
        df.set_index(_multi_index, inplace=True)

        q = Q_(df, ureg.meters)
        q.ito('cm')

        if not os.path.exists('.tmp-test'):
            os.mkdir('.tmp-test')

        filename = '.tmp-test/test.hdf5'
        h5store(filename, q.m, **{'test_col': str(q.units)})

        val, metadata = h5load(filename)

        assert_frame_equal(val, df * 100)

    def test_df_store(self):
        val, metadata = h5load(filename)
        #pint_pandas_data, m = load_as_df_qantity(f'{output_directory}/result_data_{variable}.hdf5')


if __name__ == '__main__':
    unittest.main()
