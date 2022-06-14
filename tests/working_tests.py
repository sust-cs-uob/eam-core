import unittest

import pandas as pd
from tests.directory_test_controller import get_static_path
import eam_core
from eam_core import ExcelVariable


class MyTestCase(unittest.TestCase):

    def test_RandomFunctionDataSource(self):
        # a = np.random.normal(0, 4, size=samples)
        ds = eam_core.RandomFunctionDataSource(module='numpy.random', function='normal', params=[0, 4])
        simulation_control = eam_core.SimulationControl()
        simulation_control.sample_size = 4
        simulation_control.use_time_series = False
        v = ds.get_value('test', simulation_control=simulation_control)
        print(v)

    def test_create_excel_var(self):
        variable = ExcelVariable(name='b', excel_file_name=get_static_path('data/test_data.xlsx'), sheet_name='Sheet1')
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) >= 2).all()
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) <= 4).all()

    @unittest.skip("variable has been deleted from file so this fails")
    def test_exponential(self):
        simulation_control = eam_core.SimulationControl()
        simulation_control.use_time_series = True
        simulation_control.sample_size = 3
        simulation_control.times = pd.date_range('2019-06-01', '2020-06-01', freq='MS')

        variable = ExcelVariable(name='energy_intensity_network', excel_file_name=get_static_path('data/test_data.xlsx'), sheet_name='Sheet3', times=pd.date_range('2019-06-01', '2020-06-01', freq='MS'))
        print(variable.data_source.get_value('energy_intensity_network', simulation_control=simulation_control))
        # assert (variable.data_source.get_value('energy_intensity_network', simulation_control=ngmodel.SimulationControl()) >= 2).all()

    def test_create_timeseries_excel_var(self):
        simulation_control = eam_core.SimulationControl()
        simulation_control.use_time_series = True
        simulation_control.sample_size = 3

        variable = ExcelVariable(name='b', excel_file_name=get_static_path('data/test_data.xlsx'), sheet_name='Sheet1',
                                 times=pd.date_range('2009-01-01', '2009-02-01', freq='MS'), size=10)

        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) >= 2).all()
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) <= 4).all()

if __name__ == '__main__':
    unittest.main()
