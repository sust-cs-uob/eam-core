import csv
import logging
import os
import json
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
from eam_core.util import store_dataframe, \
    store_process_metadata, store_parameter_repo_variables, get_param_names, to_target_dimension, \
    pandas_series_dict_to_dataframe, extract_functional_units

logger = logging.getLogger(__name__)


class SimulationRunner(object):
    """"
    Responsible for the execution of the simulation, including storage of results. It encapsulates the protocol of the
    simulation.

    todo: most of this isnt tested, large parts arent used. remove them?
    """

    def __init__(self, sim_control=None):
        self.sim_control = sim_control
        self.use_docker = False

    def run_SA(self, create_model_func=None, embodied=False, sim_control=None):
        """
        1. calc correlation coefficient dividing the covariance by the product of the two variables' standard deviations
        :param create_model_func:
        :type create_model_func:
        :param embodied:
        :type embodied:
        :param sim_control:
        :type sim_control:
        :return:
        :rtype:
        """

        output_directory = sim_control.output_directory
        sim_control = SimulationControl()
        sim_control.use_time_series = True
        samples = 2
        sim_control.sample_size = samples

        # evaluate model to get input variables
        model = create_model_func(sim_control=sim_control)
        model.footprint(use_phase=True, embodied=embodied, simulation_control=sim_control)

        traces: Dict[str, Dict[str, pd.Series]] = model.collect_calculation_traces()
        Y = pd.DataFrame(traces['energy'])  # .sum(axis=1)

        Y.pint.dequantify().to_hdf(f'{output_directory}/Y.h5', 'Y', table=True, mode='a')

        # logger.debug(Y.head())
        param_names = get_param_names(sim_control)
        count = len(param_names)

        idx = 0

        # calculate all relevant sets
        # single param names
        # two param names
        s = set()
        for k in param_names:
            s.add(frozenset({k}))
            for j in param_names:
                s.add(frozenset({k, j}))

        param_name_tuple = next(iter(next(iter(s))))

        logger.debug(f'param_name_tuple = {param_name_tuple}')
        # sim_control = SimulationControl()
        sim_control.param_repo.clear_cache()
        # sim_control.use_time_series = False
        # sim_control.sample_size = samples

        sim_control.single_variable_run = True
        sim_control.single_variable_names = param_name_tuple

        model = create_model_func(sim_control=sim_control)

        model.footprint(use_phase=True, embodied=embodied, simulation_control=sim_control)

        traces: Dict[str, Dict[str, pd.Series]] = model.collect_calculation_traces()

        p = traces[param_name_tuple]
        # a variable might be used in a variety of processes
        # we just take the first
        p = next(iter(p.values()))
        logger.debug(p)
        p.pint.m.to_hdf(f'{output_directory}/p.h5', param_name_tuple, table=True, mode='a')

    @staticmethod
    def get_stddev_and_mean(Y, Xi):
        Y = [i.pint.m for i in Y.values()]
        s=sum(Y)
        c=Y[0]
        std_dev = np.std(sum(Y))
        men = np.mean(sum(Y))

        return {'std_dev': std_dev,
                'mean': men,
                'cv': std_dev / men,
                'Xi_std_dev': np.std(Xi),
                'Xi_mu': np.mean(Xi),
                'Xi_cv': np.std(Xi) / np.mean(Xi),
                }
    #turns a dictionary of series into a list of arrays, removing pint units
    def dict_to_array(self,a):
        for k in a.keys():
            a[k]=a[k].pint.m
        a=list(a.values())
        for i in range(len(a)):
            a[i]=a[i].to_numpy()
        return a

    #calculates the multiple correlation between a dependent and any number of independent variables. if only one independent variable is given it returns the correlation
    def multiple_correlation(self, dep,indep):
        assert len(dep)==1
        if len(indep)>=2:
            Rxx=np.corrcoef(np.array(indep))
            Rxx=np.linalg.inv(Rxx)
            c=np.zeros(len(indep))
            for i in range(len(indep)):
                c[i]=np.corrcoef(indep[i],dep)[0,1]
            return math.sqrt(np.matmul(np.transpose(c),np.matmul(Rxx,c)))
        else:
            return np.corrcoef(indep,dep)[0,1]


    def run_OTA_SA(self, create_model_func=None, embodied=True, sim_control=None):
        """
        @todo implement http://www.real-statistics.com/correlation/basic-concepts-correlation/
        http://www.real-statistics.com/correlation/multiple-correlation/

        Perform one at a time sensitivity analysis

        (1)	Get list of variables
            ⁃	run once, get dump of variables
        (2)	For each var:
            ⁃	run the model in single mode, with all other variables fixed to mean
            ⁃	get variance/std dev of var and Y var
            ⁃	store variance value in table

        for efficiency reasons
        - uses local data
        - does not store traces
        - does not run in time series mode

        :return:
        """
        sim_control = SimulationControl()
        sim_control.use_time_series = True
        samples = 2
        sim_control.sample_size = samples

        # evaluate model to get input variables
        model = create_model_func(sim_control=sim_control)
        fp = model.footprint(use_phase=True, embodied=embodied, simulation_control=sim_control)
        cor={}
        fparray=self.dict_to_array(fp['use_phase_energy'])
        cor['all'] = SimulationRunner.multiple_correlation(self,fparray, [np.ones(len(fparray[0]))])

        param_names = get_param_names(sim_control)
        count = len(param_names)

        idx = 0

        # calculate all relevant sets
        # single param names
        # two param names
        s = set()
        for k in param_names:
            s.add(frozenset({k}))
            for j in param_names:
                s.add(frozenset({k, j}))

        # for param_name in ['num_online_TV_HH']:
        for param_name_tuple in s:

            # ignore intermediate variables
            exists = [sim_control.param_repo.exists(param_name) for param_name in param_name_tuple]
            if not all(exists):
                continue

            if any([sim_control.param_repo[param].cache is None for param in param_name_tuple]):
                # ignore parameters that were not used
                continue

            idx = idx + 1
            print(f"{idx} of {count}")
            if idx > 9:
                return model, cor
            print(f"running model for var(s) {param_name_tuple}")

            # sim_control = SimulationControl()
            sim_control.param_repo.clear_cache()
            # sim_control.use_time_series = False
            # sim_control.sample_size = samples

            sim_control.single_variable_run = True
            sim_control.single_variable_names = param_name_tuple

            model = create_model_func(sim_control=sim_control)
            fp = model.footprint(use_phase=True, embodied=embodied, simulation_control=sim_control)

            # for p in param_name_tuple:
            # param = sim_control.param_repo[p]
            # stddev_and_mean = SimulationRunner.get_stddev_and_mean(fp['use_phase_energy'], None)
            # stddevs.append(stddev_and_mean)
            # if stddev_and_mean['std_dev'] > 1:
            # variances[str(list(param_name_tuple))] = stddev_and_mean
            # print(outcome_var)

            cor[str(list(param_name_tuple))]=[self.multiple_correlation(self.dict_to_array(fp['use_phase_energy']), self.dict_to_array(fp['traces']))]

        return model, cor

    def run(self, create_model_func=None, embodied=False,
            debug=False, target_units=None, result_variables=None,
            output_persistence_config=None, functional_unit_config=None):

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

            self.store_functional_unit_value(functional_unit_config)

        self.store_json_graph()

        return self.model, self.footprint_result_dict

    def store_parameterrepository_variables(self):
        logger.info("Storing parameter_repo_variables")
        store_parameter_repo_variables(self.model, simulation_control=self.sim_control)

    def store_metadata(self):
        logger.info("Storing metadata")
        store_process_metadata(self.model, simulation_control=self.sim_control)

    def store_functional_unit_value(self, functional_unit_config=None):
        if functional_unit_config:
            if functional_unit_config.get('use_functional_units', False):
                logger.info(f'Calculating functional units...')
                f_unit_cumulative_value, f_unit_type = extract_functional_units(functional_unit_config,
                                                                                self.sim_control.param_repo.parameter_sets,
                                                                                self.sim_control.scenario)
                f_unit_json = {f_unit_type:
                               [f_unit_cumulative_value.magnitude,
                                str(f_unit_cumulative_value.units)]}
                filename = f'{self.sim_control.output_directory}/functional_unit_sum.json'

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(f_unit_json, f, ensure_ascii=False, indent=4)
                    f.close()
            else:
                logger.info('Functional units are not calculated for this model')

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
