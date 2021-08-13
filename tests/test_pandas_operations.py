import math
import unittest
import pandas as pd
import pint
import numpy as np
import pickle


def generate_series_list():
    times = pd.date_range('2020-01-01', '2020-3-01', freq='MS')
    sample_size = 3
    groups_full = ['A', 'B', 'C']

    series_length = len(times) * sample_size * len(groups_full)
    index_names_full = ['time', 'samples', 'group']

    iterables = [times, range(sample_size), groups_full]
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names_full)
    s1 = pd.Series(data=range(series_length), index=df_multi_index, name='FULL_A')

    iterables = [times, range(sample_size), groups_full[:2]]
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names_full)
    s2 = pd.Series(data=range(int(series_length * 2 / 3)), index=df_multi_index, name='TWO_COUNTRIES')

    iterables = [times, range(sample_size)]
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names_full[:2])
    s3 = pd.Series(data=range(int(series_length / 3)), index=df_multi_index, name='NO_GROUPS_A')

    iterables = [times, range(sample_size), groups_full[:1]]
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names_full)
    s4 = pd.Series(data=range(int(series_length * 1 / 3)), index=df_multi_index, name='ONE_COUNTRY')

    iterables = [times, range(sample_size)]
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names_full[:2])
    s5 = pd.Series(data=range(int(series_length / 3)), index=df_multi_index, name='NO_GROUPS_B')

    return [s1, s2, s3, s4, s5], series_length


def get_generated_series_indexing():
    times = pd.date_range('2020-01-01', '2020-3-01', freq='MS')
    sample_size = 3
    groups = ['A', 'B', 'C']
    index_names = ['time', 'samples', 'group']

    iterables = [times, range(sample_size), groups]

    return iterables, times, index_names


def get_series_and_dataframe():
    series, series_length = generate_series_list()
    iterables, times, index_names = get_generated_series_indexing()
    df_multi_index = pd.MultiIndex.from_product(iterables, names=index_names)
    df_index = pd.DataFrame(index=df_multi_index)

    for i in range(len(series)):
        df_index[series[i].name] = series[i]

    return series, df_index


def assert_lists_equal(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        assert (a[i] == b[i]) or (math.isnan(a[i]) and math.isnan(b[i]))


class DataFrameShapeTest(unittest.TestCase):
    series, series_length = generate_series_list()
    iterables, times, index_names = get_generated_series_indexing()

    def test_series_shapes(self):
        shapes = [s.shape for s in self.series]
        assert shapes == [(27,), (18,), (9,), (9,), (9,)]

    def test_rejected_dataframe(self):
        df_blank = pd.DataFrame()

        # When combining series into a dataframe, the dimensionality must not increase, or an error is thrown.
        # Thus, looking at the shape assertion above, we expect an error if the dataframe is filled in reverse.
        def fill_dataframe():
            for i in range(len(self.series)):
                r_i = len(self.series) - (i + 1)
                df_blank[self.series[r_i].name] = self.series[r_i]
        self.assertRaises(Exception, fill_dataframe)

    def test_accepted_dataframe(self):
        # When filling in order however, the first and largest series substitutes its own sufficient index
        df_blank = pd.DataFrame()
        for i in range(len(self.series)):
            df_blank[self.series[i].name] = self.series[i]
        assert df_blank.size == len(self.series) * self.series_length

    def test_accepted_indexed_dataframe(self):
        assert set(self.series[0].index.get_level_values('group').values) == {'C', 'B', 'A'}
        assert set(self.series[1].index.get_level_values('group').values) == {'B', 'A'}

        df_multi_index = pd.MultiIndex.from_product(self.iterables, names=self.index_names)
        df_index = pd.DataFrame(index=df_multi_index)

        # this dataframe has a multi-index constructed to not throw an error regardless of input order
        for i in range(len(self.series)):
            df_index[self.series[i].name] = self.series[i]
            assert df_index[self.series[i].name].size == self.series_length

        assert df_index.size == len(self.series) * self.series_length

        # show that data is duplicated when a series is brought up to a higher-dimensional dataframe
        for time_sample_index in range(self.series[2].values.max() + 1):
            for group_index in range(len(set(df_index['NO_GROUPS_A'].index.get_level_values('group').values))):
                assert df_index['NO_GROUPS_A'].values[(time_sample_index * 3) + group_index] == time_sample_index

        # show that data is filled with NaN when a series is brought into a dataframe with more same-level indices
        countries_used = set(self.series[1].index.get_level_values('group').values)
        for i in range(len(df_index['TWO_COUNTRIES'].values)):
            if ((i + 1) % 3 == 0) and i != 0:
                assert math.isnan(df_index['TWO_COUNTRIES'].values[i])
            else:
                assert not math.isnan(df_index['TWO_COUNTRIES'].values[i])

        for i in range(len(df_index['ONE_COUNTRY'].values)):
            if i % 3 == 0:
                assert not math.isnan(df_index['ONE_COUNTRY'].values[i])
            else:
                assert math.isnan(df_index['ONE_COUNTRY'].values[i])


class DataFrameOperationsTest(unittest.TestCase):
    def test_basic_operations(self):
        series, df_index = get_series_and_dataframe()

        a = (df_index['FULL_A'] + df_index['ONE_COUNTRY']).values
        b = (df_index['FULL_A'].values + df_index['ONE_COUNTRY'].values)
        assert_lists_equal(list(a), list(b))

        a = (df_index['FULL_A'] * df_index['NO_GROUPS_A']).values
        b = (df_index['FULL_A'].values * df_index['NO_GROUPS_A'].values)
        assert_lists_equal(list(a), list(b))

        a = (df_index['TWO_COUNTRIES'] / df_index['NO_GROUPS_A']).values
        b = (df_index['TWO_COUNTRIES'].values / df_index['NO_GROUPS_A'].values)
        assert_lists_equal(list(a), list(b))

        one_country_div_full = df_index['ONE_COUNTRY'] / df_index['FULL_A']
        assert len(one_country_div_full.dropna()) == len(series[3].values) - 1

        full_div_one_country = df_index['ONE_COUNTRY'] / df_index['FULL_A']
        assert len(full_div_one_country.dropna()) == len(series[3].values) - 1

    # @unittest.skip("numpy statistic functions return NaN if any element given is NaN? not good.")
    def test_statistics(self):
        series, df_index = get_series_and_dataframe()

        def drop_nan(x):
            return x[~np.isnan(x)]

        assert drop_nan(df_index['FULL_A']).max() == 26
        assert drop_nan(df_index['TWO_COUNTRIES']).max() == 17

        assert drop_nan(df_index['FULL_A']).sum() == 351
        assert drop_nan(df_index['TWO_COUNTRIES']).sum() == 153

        assert drop_nan(df_index['FULL_A']).mean() == 13.0
        assert drop_nan(df_index['TWO_COUNTRIES']).mean() == 8.5
