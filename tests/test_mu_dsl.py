import logging
import numbers
import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from pint import UnitRegistry

from eam_core import SimulationControl
from eam_core.YamlLoader import YamlLoader
from eam_core.dsl import evaluate

logger = logging.getLogger(__name__)

ureg = UnitRegistry(auto_reduce_dimensions=False)
# ureg.define('bit = [information]')
ureg.default_format = '~H'
Q_ = ureg.Quantity


class NumberTestCase(unittest.TestCase):

    def test_log(self):
        block = """
            log "Done!";
        """
        visitor = evaluate(block)

    def test_comment(self):
        block = """
            # this line is ignored
            log "Done!";
        """
        visitor = evaluate(block)

    def test_basic_int_assign(self):
        block = """
            a = 2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 2

    def test_basic_float_assign(self):
        block = """
            a = 2.2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 2.2

    def test_basic_str_assign(self):
        block = """
            a = "hello";
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == "hello"

    def test_basic_nil_assign(self):
        block = """
            a = nil;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == None

    def test_basic_add(self):
        block = """
            a = 2+2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_sub(self):
        block = """
            a = 6-2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_several_add(self):
        block = """
            a = 1 + 1 + 1 + 1;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_unary_minus(self):
        block = """
            a = -2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == -2

    def test_basic_mult(self):
        block = """
            a = 2*2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_div(self):
        block = """
            a = 8/2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_mod(self):
        block = """
            a = 14 % 5;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_pow(self):
        block = """
            a = 2^2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 4

    def test_basic_paranthesis(self):
        block = """
            a = (2+2)*2;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 8

    def test_basic_bool_true(self):
        block = """
            a = true;
        """
        visitor = evaluate(block)
        assert visitor.variables['a']

    def test_basic_bool_false(self):
        block = """
            a = false;
        """
        visitor = evaluate(block)
        assert not visitor.variables['a']

    def test_basic_bool_not(self):
        block = """
            a = !false;
        """
        visitor = evaluate(block)
        assert visitor.variables['a']

    def test_basic_eq(self):
        block = """
            a = 3 == 1;
        """
        visitor = evaluate(block)
        assert not visitor.variables['a']

    def test_basic_gt(self):
        block = """
            a = 3 > 1;
        """
        visitor = evaluate(block)
        assert visitor.variables['a']

    def test_basic_get(self):
        block = """
            a = 3 >= 1;
            b = 3 >= 3;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b']

    def test_basic_let(self):
        block = """
            a = 1 <= 1;
            b = 1 <= 3;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b']

    def test_basic_lt(self):
        block = """
            a = 1 < 3;
        """
        visitor = evaluate(block)
        assert visitor.variables['a']

    def test_basic_and(self):
        block = """
            a = true && true;
            b = (2<3) &&  ( 4 >= 4);
            c = false && true;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b'] and not visitor.variables['c']

    def test_basic_or(self):
        block = """
            a = true || false;
            b = (2 <3 ) ||  ( 4 >= 1);
            c = false || false;
        """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b'] and not visitor.variables['c']

    def test_basic_if(self):
        block = """
                   a = true ;
                   if (a){
                        b = 2;
                   }
               """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b']

    def test_basic_if_non_zero_is_true(self):
        block = """
                   a = 10000 ;
                   if (a){
                        b = 2;
                   }
               """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 10000 and visitor.variables['b'] == 2

    def test_basic_if_zero_is_false(self):
        block = """
                   a = 0 ;
                   if (a){
                        b = 2;
                   }
               """
        visitor = evaluate(block)

        assert visitor.variables['a'] == 0 and not 'b' in visitor.variables

    def test_basic_if_else_if_else(self):
        block = """
                   a = 0 ;
                   b = 0 ;
                   if (a != 0){
                        b = 2;
                   } else if (b != 0){
                        c = 1;
                   } else {
                        c = 2;
                   }
               """
        visitor = evaluate(block)
        assert visitor.variables['c'] == 2

    def test_basic_if_else_if(self):
        block = """
                   a = 0 ;
                   b = 1 ;
                   if (a){
                        b = 2;
                   } else if (b){
                        c = 1;
                   }
               """
        visitor = evaluate(block)
        assert not visitor.variables['a'] and visitor.variables['b'] and visitor.variables['c'] == 1

    def test_basic_if_else(self):
        block = """
                   a = true ;
                   if (a){
                        b = 2;
                   } else {
                        c = false;
                   }
               """
        visitor = evaluate(block)
        assert visitor.variables['a'] and visitor.variables['b'] and 'c' not in visitor.variables

    def test_basic_while(self):
        block = """
                   a = 10;
                   b = 0;
                   while (a > 0){
                        b= b+1;
                        a = a- 1;
                   }
               """
        visitor = evaluate(block)
        assert visitor.variables['a'] == 0 and visitor.variables['b'] == 10


class PandasTestCase(unittest.TestCase):

    def test_multi_line_variables_with_objects(self):
        line = """
        b = a * 2.
        c = a * 2.
        return a
        """

        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        visitor = evaluate(line, variables=variables)

        assert_frame_equal(visitor.variables['b'], visitor.variables['c'])
        data = {'col1': [2., 4.], 'col2': [6., 8.]}
        dfx2 = pd.DataFrame(data=data)
        assert_frame_equal(visitor.variables['b'], dfx2)

        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        assert_frame_equal(visitor.variables['a'], df)

    def test_pandas_not_real(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        assert not isinstance(df, numbers.Real)

    def test_pandas_div(self):
        d = {'col1': [2., 2.], 'col2': [4., 4.]}
        df = pd.DataFrame(data=d)

        line = 'a = b/2'
        variables = {}
        variables['b'] = df

        visitor = evaluate(line, variables=variables)

        d = {'col1': [1., 1.], 'col2': [2., 2.]}
        df_new = pd.DataFrame(data=d)

        assert_frame_equal(visitor.variables['a'], df_new)

    def test_pandas_add(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a+2;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df + 2)

    def test_pandas_sub(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a-2;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df - 2)

    def test_pandas_m_add(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a + 1 + 1 + 1;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df + 1 + 1 + 1)

    def test_basic_unary_minus(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = -a;
        """
        visitor = evaluate(block, variables=variables)
        assert_frame_equal(visitor.variables['b'], -df)

    def test_basic_mod(self):
        block = """
            a = df % 5;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [14.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [4.]}))

    def test_basic_pow(self):
        block = """
            a = df^2;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [14.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [196.]}))

    def test_basic_paranthesis(self):
        block = """
            a = (df+2)*2;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [8.]}))

    def test_basic_gt(self):
        block = """
            a = df > 1;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == True

    def test_basic_eq(self):
        block = """
            a = df == 1;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [1., 1.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == True

    def test_pandas_test_zero(self):
        block = """
            a = df == 0;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 1.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == False

    def test_log_pandas_test_zero(self):
        block = """
            log df == 0;
        """
        evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 1.]})})

    def test_basic_if_series(self):
        block = """
                   if (0 == s) and (s == 0) {
                        b = s;
                   }
               """
        visitor = evaluate(block, variables={'s': pd.Series([0., 0.])})
        print(visitor.variables['b'])
        assert_series_equal(visitor.variables['b'], pd.Series([0., 0.]))

    def test_basic_if_df(self):
        block = """
                   if (0 == df) and (df == 0) {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 0.]})})
        print(visitor.variables['b'])
        assert_frame_equal(visitor.variables['b'], pd.DataFrame(data={'c': [0., 0.]}))

    def test_basic_and(self):
        block = """
            a = (df<3) &&  ( 4 >= 4);

        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a']

    def test_basic_or(self):
        block = """

            a = (df < 3 ) ||  ( 4 >= 1);

        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a']


class QuantityTestCase(unittest.TestCase):

    def test_multi_line_variables_with_objects(self):
        line = """
        b = a * 2.
        c = a * 2.
        return a
        """

        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)
        quantity = Q_(df, 'm')

        variables = {}
        variables['a'] = quantity

        visitor = evaluate(line, variables=variables)

        assert_frame_equal(visitor.variables['b'].m, visitor.variables['c'].m)
        data = {'col1': [2., 4.], 'col2': [6., 8.]}
        dfx2 = pd.DataFrame(data=data)
        quantityx2 = Q_(dfx2, 'm')
        assert_frame_equal(visitor.variables['b'].m, quantityx2.m)

        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)
        quantity3 = Q_(df, 'm')

        assert_frame_equal(visitor.variables['a'].m, quantity3.m)

    def test_pandas_not_real(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        assert not isinstance(df, numbers.Real)

    def test_pandas_div(self):
        d = {'col1': [2., 2.], 'col2': [4., 4.]}
        df = pd.DataFrame(data=d)

        line = 'a = b/2'
        variables = {}
        variables['b'] = df

        visitor = evaluate(line, variables=variables)

        d = {'col1': [1., 1.], 'col2': [2., 2.]}
        df_new = pd.DataFrame(data=d)

        assert_frame_equal(visitor.variables['a'], df_new)

    def test_pandas_add(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a+2;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df + 2)

    def test_pandas_sub(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a-2;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df - 2)

    def test_pandas_m_add(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = a + 1 + 1 + 1;
        """
        visitor = evaluate(block, variables=variables)

        assert_frame_equal(visitor.variables['b'], df + 1 + 1 + 1)

    def test_basic_unary_minus(self):
        data = {'col1': [1., 2.], 'col2': [3., 4.]}
        df = pd.DataFrame(data=data)

        variables = {}
        variables['a'] = df

        block = """
            b = -a;
        """
        visitor = evaluate(block, variables=variables)
        assert_frame_equal(visitor.variables['b'], -df)

    def test_basic_mod(self):
        block = """
            a = df % 5;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [14.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [4.]}))

    def test_basic_pow(self):
        block = """
            a = df^2;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [14.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [196.]}))

    def test_basic_paranthesis(self):
        block = """
            a = (df+2)*2;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        assert_frame_equal(visitor.variables['a'], pd.DataFrame(data={'c': [8.]}))

    def test_basic_gt(self):
        block = """
            a = df > 1;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == True

    def test_basic_eq(self):
        block = """
            a = df == 1;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [1., 1.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == True

    def test_pandas_test_zero(self):
        block = """
            a = df == 0;
        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 1.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a'] == False

    def test_log_pandas_test_zero(self):
        block = """
            log df == 0;
        """
        evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 1.]})})

    def test_basic_if_series(self):
        block = """
                   if (0 == s) and (s == 0) {
                        b = s;
                   }
               """
        visitor = evaluate(block, variables={'s': pd.Series([0., 0.])})
        print(visitor.variables['b'])
        assert_series_equal(visitor.variables['b'], pd.Series([0., 0.]))

    def test_short_if_series_false(self):
        block = """
                   if (s) {
                        b = 1;
                   } else {
                        b = s;
                   }
               """
        visitor = evaluate(block, variables={'s': pd.Series([0., 0.])})
        print(visitor.variables['b'])
        assert_series_equal(visitor.variables['b'], pd.Series([0., 0.]))

    def test_short_if_series_true(self):
        block = """
                   if (s) {
                        b = 1;
                   } else {
                        b = s;
                   }
               """
        visitor = evaluate(block, variables={'s': pd.Series([2., 1.])})
        print(visitor.variables['b'])
        assert visitor.variables['b'], 1

    def test_basic_if_df(self):
        block = """
                   if (0 == df) and (df == 0) {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 0.]})})
        print(visitor.variables['b'])
        assert_frame_equal(visitor.variables['b'], pd.DataFrame(data={'c': [0., 0.]}))

    def test_short_if_df_true(self):
        block = """
                   if (df) {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2., 1.]})})
        print(visitor.variables['b'])
        assert_frame_equal(visitor.variables['b'], pd.DataFrame(data={'c': [2., 1.]}))

    def test_short_if_df_false(self):
        block = """
                   if (df) {
                        b = 2;
                   } else {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [0., 0.]})})
        print(visitor.variables['b'])
        assert visitor.variables['b'] == 2

    def test_basic_and(self):
        block = """
            a = (df<3) &&  ( 4 >= 4);

        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a']

    def test_basic_or(self):
        block = """

            a = (df < 3 ) ||  ( 4 >= 1);

        """
        visitor = evaluate(block, variables={'df': pd.DataFrame(data={'c': [2.]})})
        print(visitor.variables['a'])
        assert visitor.variables['a']

    def test_pint_pandas(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d, dtype='pint[W]')

        line = 'a = b * 2'
        visitor = evaluate(line, variables={'b': df})

        from pandas.util.testing import assert_frame_equal

        d = {'col1': [2., 4.], 'col2': [6., 8.]}
        df_new = pd.DataFrame(data=d, dtype='pint[W]')

        assert_frame_equal(visitor.variables['a'], df_new)
        assert visitor.variables['a']['col1'].pint.units == 'watt'

    def test_pint_pandas_df_basic_if_true(self):
        d = {'col1': [1, 2]}
        df = pd.DataFrame(data=d, dtype='pint[W]')

        block = """
                   if (df) {
                        b = 2;
                   } else {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': df})
        print(visitor.variables['b'])
        assert visitor.variables['b'], 2

    def test_pint_pandas_df_basic_if_false(self):
        d = {'col1': [0, 0]}
        df = pd.DataFrame(data=d, dtype='pint[W]')

        block = """
                   if (df) {
                        b = df;
                   } else {
                        b = 2;
                   }
               """
        visitor = evaluate(block, variables={'df': df})
        print(visitor.variables['b'])
        assert visitor.variables['b'], 2

    def test_pint_pandas_series_basic_if_true(self):
        d = {'col1': [1, 2]}
        df = pd.Series(data=d, dtype='pint[W]')

        block = """
                   if (df) {
                        b = 2;
                   } else {
                        b = df;
                   }
               """
        visitor = evaluate(block, variables={'df': df})
        print(visitor.variables['b'])
        assert visitor.variables['b'] == 2

    def test_pint_pandas_series_basic_if_false(self):
        d = {'col1': [0, 0]}
        df = pd.Series(data=d, dtype='pint[W]')

        block = """
                   if (df) {
                        b = df;
                   } else {
                        b = 2;
                   }
               """
        visitor = evaluate(block, variables={'df': df})
        print(visitor.variables['b'])
        assert visitor.variables['b'] == 2

    def test_pint_pandas_timeseries(self):
        start_date = '2010-01-01'
        end_date = '2030-12-01'
        date_range = pd.date_range(start_date, end_date, freq='MS')
        samples = 3
        iterables = [date_range, range(samples)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)

        power = pd.Series(np.full((len(date_range), samples), 5).ravel(), index=_multi_index, dtype='pint[W]')

        time = pd.Series(np.full((len(date_range), samples), 5).ravel(), index=_multi_index, dtype='pint[s]')

        line = 'energy = power * time'
        visitor = evaluate(line, variables={'power': power, 'time': time})

        print(visitor.variables['energy'].shape)
        print(visitor.variables['energy'].pint.units)
        assert visitor.variables['energy'].iloc[0].m == 25
        assert visitor.variables['energy'].pint.to('Wh').iloc[0].m == 0.006944444444444444

    def test_pint_pandas_timeseries_mix_var(self):
        start_date = '2010-01-01'
        end_date = '2030-12-01'
        date_range = pd.date_range(start_date, end_date, freq='MS')
        samples = 3
        iterables = [date_range, range(samples)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)

        power = pd.Series(np.full((len(date_range), samples), 5).ravel(), index=_multi_index, dtype='pint[W]')
        channels = pd.Series(np.full((len(date_range), samples), 5).ravel(), index=_multi_index,
                             dtype='pint[dimensionless]')

        sim_control = SimulationControl()
        sim_control.times = date_range
        sim_control.sample_size = samples
        sim_control.use_time_series = True
        sim_control._df_multi_index = _multi_index

        time = YamlLoader().create_variable('ref_duration', sim_control, '2628000s').data_source.get_value(
            'ref_duration', sim_control)

        block = '''
        aggreate_power = mean_power_per_linear_channel * number_of_BBC_linear_channels

        energy = aggreate_power * ref_duration

        return energy
        '''
        visitor = evaluate(block,
                           variables={'mean_power_per_linear_channel': power, 'number_of_BBC_linear_channels': channels,
                                      'ref_duration': time})

        print(visitor.variables['energy'].shape)
        print(visitor.variables['energy'].pint.units)
        # assert visitor.variables['energy'].iloc[0].m == 25
        # assert visitor.variables['energy'].pint.to('Wh').iloc[0].m == 0.006944444444444444


if __name__ == '__main__':
    unittest.main()
