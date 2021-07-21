import unittest
import pandas as pd
import eam_core
from eam_core import ExcelVariable
from tests.directory_test_controller import get_static_path

class MyTestCase(unittest.TestCase):

    # works
    def test_RandomFunctionDataSource(self):
        # a = np.random.normal(0, 4, size=samples)
        ds = eam_core.RandomFunctionDataSource(module='numpy.random', function='normal', params=[0, 4])
        simulation_control = eam_core.SimulationControl()
        simulation_control.sample_size = 4
        simulation_control.use_time_series = False
        v = ds.get_value('test', simulation_control=simulation_control)
        print(v)

    @unittest.skip("not sure RandomFunctionDataSource is still needed")
    def test_timeseries_RandomFunctionDataSource(self):
        ds = eam_core.RandomFunctionDataSource(module='numpy.random', function='normal', params=[.0, 4.])
        simulation_control = eam_core.SimulationControl()
        simulation_control.sample_size = 4
        simulation_control.use_time_series = True
        v = ds.get_value('test', simulation_control=simulation_control)
        print(v)

    @unittest.skip("not sure RandomFunctionDataSource is still needed")
    def test_timeseries_mean_RandomFunctionDataSource(self):
        ds = eam_core.RandomFunctionDataSource(module='numpy.random', function='normal', params=[.0, 4.])
        simulation_control = eam_core.SimulationControl()
        simulation_control.sample_size = 4
        simulation_control.use_time_series = True
        simulation_control.sample_mean_value = True
        v = ds.get_value('test', simulation_control=simulation_control)
        print(v)

    # works
    def test_create_excel_var(self):
        variable = ExcelVariable(name='b', excel_file_name=get_static_path('data/test_data.xlsx'), sheet_name='Sheet1')
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) >= 2).all()
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) <= 4).all()

    # works
    def test_create_timeseries_excel_var(self):
        simulation_control = eam_core.SimulationControl()
        simulation_control.use_time_series = True
        simulation_control.sample_size = 3

        variable = ExcelVariable(name='b', excel_file_name='tests/data/test_data.xlsx', sheet_name='Sheet1',
                                 times=pd.date_range('2009-01-01', '2009-02-01', freq='MS'), size=10)

        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) >= 2).all()
        assert (variable.data_source.get_value('b', simulation_control=eam_core.SimulationControl()) <= 4).all()


if __name__ == '__main__':
    unittest.main()
