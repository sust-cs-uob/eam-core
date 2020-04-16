import unittest

import numpy as np
import pandas as pd
import pint

from eam_core import SimulationControl
from eam_core import ureg, Q_
from eam_core.YamlLoader import YamlLoader
from eam_core.util import quantity_dict_to_dataframe


class MyTestCase(unittest.TestCase):

    # def test_data_vol_from_bitrate_and_time(self):
    #     Q_('1 Mbps') * Q_('1 minute') == Q_('60 Mbit')

    # def tets_data_energy_context(self):
    #     c = pint.Context('energy')
    #     c.add_transformation('[energy]', '[time]', lambda ureg, x: ureg.speed_of_light / x)
    #     ureg.add_context(c)
    #
    #     energy = Q_('1 J / b') * Q_('1 b')
    #     print(energy)
    #     assert energy == Q_('1 J')

    def test_inline_conversion(self):
        assert Q_('1 Mbps') * Q_('1 minute') == Q_('60 Mbit')

    def test_jpb_default_unit(self):
        energy = Q_('1 J / bit') * Q_('1 bit')
        print(energy)
        assert energy == Q_('1 J')

    def test_jpb_scientific_unit(self):
        energy = Q_('1.61E-07 J/bit') * Q_('2.45E+19 bit')
        print(energy.to(ureg.watt_hour).to_compact())

    def test_dataframe_quantity_parsing(self):
        print(Q_('0minute'))
        print(Q_('1minute/day'))
        print(Q_('1minute/day') * Q_('30day/month'))

    def test_dataframe_quantity_mult(self):
        d = {'col1': [2., 2.], 'col2': [4., 4.]}
        df = pd.DataFrame(data=d)

        distance = Q_(df, 'centimeter')
        distance *= 2
        print(distance.to(ureg.meter))

    def test_dataframe_quantity_add(self):
        d = {'col1': [2., 2.], 'col2': [4., 4.]}
        df = pd.DataFrame(data=d)

        distance = Q_(df, 'centimeter')
        distance += distance
        print(distance.to(ureg.meter))

        # @todo review return values from DSL
    @unittest.skip("old DSL. New does not return values - need review")
    def test_simple_model_with_units(self):
        doc = u"""
        Processes:        
          - name: a
            formula:
              text: |
               return power * time
            input_variables:
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1                
                variables:
                  # - time
                  - power
              - type: StaticVariables
                variables:                  
                  time: 10second

        Metadata:
          model_name: test model
          file_locations:
            - file_alias: test_location
              file_name: tests/data/test_data.xlsx            
        """

        yaml_structure = YamlLoader.load_definitions(doc)

        sim_control = SimulationControl()
        # sim_control.sample_size = 1
        # sim_control.sample_mean_value = True

        sim_control.use_time_series = True
        sim_control.times = pd.date_range('2016-01-01', '2017-01-01', freq='MS')
        sim_control.sample_size = 2
        sim_control.sample_mean_value = True

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)

        print(type(fp['use_phase_energy']['a'].m))
        assert isinstance(fp['use_phase_energy']['a'].m, pd.Series)

        assert np.allclose(fp['use_phase_energy']['a'], [20.], rtol=.1)

    def test_keep_datatype(self):
        x = 2
        values = np.full((len(pd.date_range('2016-01-01', '2016-02-01', freq='MS')), 2), x)

        df = pd.DataFrame(values.ravel())

        iterables = [pd.date_range('2016-01-01', '2016-02-01', freq='MS'), range(0, 2)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)

        df.set_index(_multi_index, inplace=True)

        d = df
        assert d.index.names == ['time', 'samples']

        e = Q_(d, ureg.meters)
        assert e.m.index.names == ['time', 'samples']

        c = e.copy()
        assert e.m.index.names == ['time', 'samples']
        e.mean()
        assert e.m.index.names == ['time', 'samples']

    def test_multi_col_df_quantity(self):
        q_dict = {}
        samples = 2
        date_range = pd.date_range('2016-01-01', '2016-12-01', freq='MS')

        df = pd.DataFrame(data=np.arange(len(date_range) * samples).ravel(), columns=['a'])
        iterables = [date_range, range(samples)]
        index_names = ['time', 'samples']
        _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)
        df.set_index(_multi_index, inplace=True)

        df_b = pd.DataFrame(data=np.arange(len(date_range) * samples).ravel(), columns=['b'])
        df_b.set_index(_multi_index, inplace=True)

        df['b'] = df_b
        e = Q_(df, ureg.kg)
        print(e.to('gram'))


if __name__ == '__main__':
    unittest.main()
