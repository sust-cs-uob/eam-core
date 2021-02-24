import csv
import logging
import os
import pint
from logging.config import dictConfig
from typing import Dict

# matplotlib.use('Agg')
import numpy as np
import pandas as pd
import simplejson
import math
import yaml
from pip._vendor.pkg_resources import resource_filename, Requirement

from eam_core import SimulationControl

logger = logging.getLogger(__name__)

from eam_core.util import store_process_metadata, store_dataframe, to_target_dimension, store_parameter_repo_variables

class SimulationRunner(object):
    """"
    Responsible for the execution of the simulation, including storage of results. It encapsulates the protocol of the
    simulation.


    """

    def __init__(self, sim_control=None):
        self.sim_control = sim_control
        self.use_docker = False

    def run(self, create_model_func=None, embodied=False,
            pickle_average=True, debug=False, target_units=None, result_variables=None, output_persistence_config=None):

        self.model = create_model_func(sim_control=self.sim_control)

        logger.info("Evaluating model")
        self.footprint_result_dict = self.model.footprint(use_phase=True, embodied=embodied,
                                                          simulation_control=self.sim_control,
                                                          debug=debug)

        if output_persistence_config is None or not output_persistence_config:
            return self.model, self.footprint_result_dict

        self.store_metadata()

        if output_persistence_config.get('process_traces'):
            self.store_process_traces(output_persistence_config.get('process_traces'), self.sim_control)

        if output_persistence_config.get('store_traces', True):
            self.store_process_variables(result_variables, target_units)
            # self.store_process_input_variables(target_units)

            self.store_input_var_csv(target_units)

            self.store_parameterrepository_variables()

        self.store_json_graph()

        return self.model, self.footprint_result_dict

    def store_parameterrepository_variables(self):
        logger.info("Storing parameter_repo_variables")
        store_parameter_repo_variables(self.model, simulation_control=self.sim_control)

    def store_metadata(self):
        logger.info("Storing metadata")
        store_process_metadata(self.model, simulation_control=self.sim_control)

    def store_input_var_csv(self, target_units):
        all_vars = self.model.collect_input_variables()
        with open(f'{self.sim_control.output_directory}/input_vars.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['process name', 'variable name', 'mean value', 'unit'])
            for proc_name, vars in sorted(all_vars.items()):
                for var_name, var in sorted(vars.items()):
                    val = to_target_dimension(var_name, var, target_units)
                    mean = val.pint.m.mean()
                    writer.writerow((proc_name, var_name, mean, str(val.pint.units)))

    def store_json_graph(self):
        logger.info("Storing calculation graph")
        with open(f'{self.sim_control.output_directory}/process_result_graph.json', 'w') as f:
            f.write(simplejson.dumps(self.sim_control.trace, default=lambda obj: str(obj)))

    def get_process_variable_values(self, variable_name, target_units=None, simulation_control=None):
        traces = self.model.collect_calculation_traces()
        _variable_values = traces[variable_name]
        df, metadata = pandas_series_dict_to_dataframe(_variable_values, target_units=target_units,
                                                       var_name=variable_name,
                                                       simulation_control=simulation_control)
        return df, metadata

    def store_process_traces(self, process_traces, simulation_control):
        for process, data in self.model.process_graph.nodes(data=True):
            logger.debug(f'collecting calculation trace for process {process.name}')

            if process.name in process_traces:
                df, metadata = pandas_series_dict_to_dataframe(process._DSL_variables,
                                                               simulation_control=simulation_control)
                logger.info(f'metadata is {metadata}')
                subdirectory = ''
                filename = f'{simulation_control.output_directory}/{subdirectory}/process_trace_{process}_{simulation_control.model_run_datetime}.xlsx'

                if not os.path.exists(f'{simulation_control.output_directory}/{subdirectory}'):
                    os.mkdir(f'{simulation_control.output_directory}/{subdirectory}')

                df = self.to_reduced_units(df.pint).pint.dequantify()
                # df = self.to_reduced_units(df.pint)
                # df = df.pint.to_base_units().pint.dequantify()
                # df = df.pint.dequantify()
                # df.columns = df.columns.droplevel(1)
                writer = pd.ExcelWriter(filename)
                df.to_excel(writer)
                writer.close()

    def to_reduced_units(self, pint):
        obj = pint._obj
        df = pint._obj
        index = object.__getattribute__(obj, 'index')
        # name = object.__getattribute__(obj, '_name')
        return pd.DataFrame({
            col: df[col].pint.to_reduced_units()
            for col in df.columns
        }, index=index)

    def store_process_variables(self, variable_names, target_units):
        """
        Given a list of variable names, stores these variables for all model processes that use this variable.
        :param variable_names:
        :type variable_names:
        :param target_units:
        :type target_units:
        :return:
        :rtype:
        """
        traces: Dict[str, Dict[str, pd.Series]] = self.model.collect_calculation_traces()

        for variable_name in variable_names:
            logger.info(f'storing results for variable {variable_name}')
            _variable_values: Dict[str, pd.Series] = traces[variable_name]

            store_dataframe(_variable_values, simulation_control=self.sim_control,
                            target_units=target_units, variable_name=variable_name)

    def store_process_input_variables(self, target_units):
        """
        Given a list of variable names, stores these variables for all model processes that use this variable.
        :param variable_names:
        :type variable_names:
        :param target_units:
        :type target_units:
        :return:
        :rtype:
        """
        process_variables = self.model.collect_input_variables()

        var_dict = {}
        for proc_name, proc_data in process_variables.items():
            for var_name, val in proc_data.items():
                var_dict[var_name] = val

        var_df = pd.DataFrame(var_dict, index=val.index)

        store_dataframe(var_df, simulation_control=self.sim_control, target_units=target_units,
                        variable_name='input_vars')
