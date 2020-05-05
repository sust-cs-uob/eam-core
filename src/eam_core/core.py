# -*- coding: utf-8 -*-
from functools import partial

from pint import UnitRegistry

from eam_core import dsl
from eam_core.dsl import EvalVisitor

ureg = UnitRegistry(auto_reduce_dimensions=False)
Q_ = ureg.Quantity

__author__ = "Daniel Schien"

import sys
from abc import abstractmethod
from collections import defaultdict
from typing import List, Any, Tuple, Dict

import copy

import networkx as nx
import numpy as np
import pandas as pd
from table_data_reader import ParameterRepository, TableParameterLoader, DistributionFunctionGenerator, \
    GrowthTimeSeriesGenerator
from networkx import DiGraph, ancestors
from networkx.readwrite import json_graph

import logging

import eam_core.log_configuration as logconf
logconf.config_logging()

logger = logging.getLogger(__name__)

CARBON_INTENSITY_kg_kWh = 0.41205  # kg/kWh
J_p_kWh = 3.60E+06
kWh_p_J = 2.7778e-7
# CARBON_INTENSITY_g_J = CARBON_INTENSITY_kg_kWh / J_p_kWh * 1000  # g/J
days_per_month = 365 / 12

metrics = ['use_phase_energy', 'use_phase_carbon', 'embodied_carbon']


class SimulationControl(object):
    """
    Global simulation settings and control switches.
    These are here to control the settings that effect the result values of the simulation

    sample_size = 100

    sample_mean_value - if True, all sample elements in all variables will have the same, mean value

    use_time_series - if True, variables will have a time axis
    #
    times = the time range for the variables



    """

    def __init__(self):
        self.scenario = 'default'
        self.output_directory = None
        self._df_multi_index = None
        self.trace = None
        self.index_names = ['time', 'samples']
        self.cache = defaultdict(dict)
        self.param_repo = ParameterRepository()
        self.use_time_series = False
        self.sample_mean_value = False
        self.single_variable_names = None
        self.single_variable_run = False
        self.excel_handler = 'openpyxl'
        self.times = pd.date_range('2009-01-01', '2017-01-01', freq='MS')
        self.sample_size = 100
        self.with_pint_units = True
        self.process_ids = False
        self.variable_ids = False
        self.filename = None

    def reset(self):
        logger.info("Resetting Simulation Control Parameters")
        self.__init__()

    def get_variable_value(self, variable: 'Variable', process):
        """
        We cache variables for single_variable_run

        :param process:
        :param variable:
        :return:
        """

        if not variable.name in self.cache:
            self.cache[process.name][variable.name] = variable.data_source.get_value(variable.name, self,
                                                                                     **{'process_name': process.name})
        return self.cache[process.name][variable.name]


def klass_from_obj_type(cm_json: Dict) -> Any:
    """Get reference to class (n.b. not an instance) given the value for a key 'obj_type' in the given json dict"""
    module_name, klass_path = cm_json['obj_type'].rsplit('.', 1)
    # @todo make this load from file if not available: http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    klass = getattr(sys.modules[module_name], klass_path)
    return klass


def np_to_json(dict_obj) -> dict:
    for k, v in dict_obj.items():
        # if v is numpy data http://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
        if type(v).__module__ == np.__name__:
            dict_obj.update({k: v.tolist()})
    return dict_obj


# define a new metaclass which overrides the "__call__" function
class PostInitClass(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass() """
        obj = type.__call__(cls, *args, **kwargs)
        obj.post_init(*args, **kwargs)
        return obj


def gen_static_multi_index_df(val, name, settings, distribution=None, unit=None):
    times = settings['times']
    samples = settings['sample_size']

    iterables = [times, range(0, samples)]
    index_names = settings['index_names']
    _multi_index = pd.MultiIndex.from_product(iterables, names=index_names)

    if distribution is None:
        distribution = np.full((len(times), samples), val)
    # else:
    #     assert distribution.shape == (len(times) * samples,)
    #
    # if not unit:
    #     df = pd.DataFrame(distribution.ravel())
    # else:
    #     df = pd.DataFrame(distribution.ravel(), dtype=f'pint[{unit}]')
    #
    # df.set_index(_multi_index, inplace=True)
    # df.columns = [name]
    #
    if not unit or str(unit) == '':

        series = pd.Series(distribution.ravel(), index=_multi_index, dtype=f'pint[dimensionless]')
    else:
        series = pd.Series(distribution.ravel(), index=_multi_index, dtype=f'pint[{unit}]')

    return series


class DataSource(object):
    @abstractmethod
    def get_value(self, name, simulation_control, **kwargs):
        pass


class FinalDataSource(DataSource):
    """
    A data source that is static and always returns the same value.
    Like scalar, except that it does not resize the value provided.

    @todo remove one of them...
    """
    value: [float]

    def get_value(self, name, simulation_control, **kwargs):
        return self.value

    @classmethod
    def create(cls, value):
        inst = cls()
        inst.value = value
        return inst


class ScalarDataSource(DataSource):
    """
    Resizes a scalar value to the required sample size for the MC runs.
    """
    scalar: float

    def __init__(self, scalar):
        self.scalar = scalar

    def get_value(self, name, simulation_control, **kwargs):
        if simulation_control.use_time_series:
            return gen_static_multi_index_df(self.scalar, name, simulation_control.__dict__)

        return np.full(simulation_control.sample_size, self.scalar)

    def __str__(self):
        return f"Scalar: {self.scalar}"


class RandomFunctionDataSource(DataSource):
    """
    Backed by a random number generator.

    """

    def __init__(self, module, function, params):
        self.params = params
        self.function = function
        self.module = module

    def get_value(self, name, simulation_control, **kwargs):
        params = dict(zip(['param_a', 'param_b', 'param_c'], self.params))
        params.update(simulation_control.__dict__)

        if simulation_control.use_time_series:
            # cagr = None, times = None, size = None, index_names = None, ref_date = None
            generator = GrowthTimeSeriesGenerator(module_name=self.module, distribution_name=self.function,
                                                  size=simulation_control.sample_size, **params)
        else:
            # {'param_a': self.params[0],'param_b': self.params[1] if }
            generator = DistributionFunctionGenerator(module_name=self.module, distribution_name=self.function,
                                                      size=simulation_control.sample_size, **params)
        return generator.generate_values(name=name)


class ExcelDataSource(DataSource):
    file_name: str
    sheet_name: str
    variable_name: str

    def __init__(self, file_name=None, sheet_name=None, variable_name=None):
        self.variable_name = variable_name
        self.sheet_name = sheet_name
        self.file_name = file_name

    def __str__(self):
        return f'Excel Source: {self.variable_name}'

    def get_value(self, name, simulation_control, **kwargs):
        logger.debug(f'sampling variable {self.variable_name} for process {name}')
        param_repo = simulation_control.param_repo

        if not param_repo.exists(self.variable_name):
            logger.debug('opening excel file')
            loader = TableParameterLoader(filename=self.file_name, table_handler=simulation_control.excel_handler)
            loader.load_into_repo(repository=param_repo, id_flag=simulation_control.variable_ids)

        param = param_repo.get_parameter(self.variable_name, scenario_name=simulation_control.scenario)
        if 'process_name' in kwargs:
            param.add_usage(kwargs['process_name'], name)

        temp_sim_config_copy = copy.copy(simulation_control.__dict__)
        if temp_sim_config_copy['single_variable_run'] and not self.variable_name in temp_sim_config_copy[
            'single_variable_names']:
            # during single_variable_run all variables are mean values, except for the single var
            temp_sim_config_copy['sample_mean_value'] = True
        parameter = param(temp_sim_config_copy)
        # assert quantity.m.index.names == ['time', 'samples']
        # quantity = Q_(parameter, param.unit)
        # assert quantity.m.index.names == ['time', 'samples']
        return parameter

    # UI representation methods.


class Variable(metaclass=PostInitClass):
    """
    A variable has a name and a value.
    The value will be set as an attribute to a process model instance at model evaluation time.
    """

    # default fields
    name: str
    data_source: DataSource

    @classmethod
    def static_variable(cls, name, value):
        """
        A variable that is initialised with the value given at instantiation time.
        This is different to a scalar variable that will extrapolate/fill a variable to fit time series dimensions from
        a scalar value
        :param name:
        :param value:
        :return:
        """
        ds = FinalDataSource.create(value=value)
        var = cls(name=name, data_source=ds)
        return var

    def __init__(self, *args, name=None, data_source=None, is_global: bool = False, **kwargs):
        # Req Var 0. Variables can be marked global if they should be made available to all processes in the service model
        self.is_global = is_global
        self.name = name

        if data_source is not None:
            self.data_source: DataSource = data_source

    def __str__(self):
        return f'{self.__class__} {self.name}'

    def post_init(self, *args, **kwargs):
        """hook for inheritance compatible post init actions"""

    @staticmethod
    def from_json_hook(**kwargs) -> Dict:
        """Pre-Process json data before instantiating Variables"""
        return kwargs

    @classmethod
    def from_json(cls, json) -> 'Variable':
        name = json["name"]
        data_source_json = json["data_source"]
        data_source_class = klass_from_obj_type(data_source_json)
        data_source_fields = data_source_json["fields"]
        data_source = data_source_class(**data_source_fields)

        return Variable(name=name, data_source=data_source)


class ExcelVariable(Variable):
    """A variable backed by a definition in an excel spreadsheet.
    """

    def __init__(self, name=None, sheet_name=None, excel_file_name=None, excel_var_name=None, *args, **kwargs):
        """
        Instantiate this variable and add to the spreadsheet cache
        :param kwargs: sheet_name, excel_file_name
        """

        super().__init__(*args, name=name, **kwargs)
        self.name = name
        excel_var_name = excel_var_name if excel_var_name else name
        excel_file_name = excel_file_name
        sheet_name = sheet_name

        logger.debug(
            f"Creating new variable {self.name} from with excel data source for {excel_file_name} variable {excel_var_name}")

        self.data_source = ExcelDataSource(file_name=excel_file_name, sheet_name=sheet_name,
                                           variable_name=excel_var_name)


class Formula(object):

    def __init__(self, text):
        self.text = text


class FormulaModel(object):

    def __init__(self, formula: Formula, ):
        self.formula = formula
        self.variable_cache = {}

    def evaluate(self, sim_control, input_variables: Dict[str, Variable] = None,
                 export_variable_names: Dict[str, str] = None,
                 DSL_variable_dict=None, **kwargs) -> \
        Tuple[Dict[str, Variable], EvalVisitor]:
        """
        1. variables and add to DSL
        2. evaluate
        3. store result variables

        :param sim_control:
        :type sim_control:
        :param input_variables: names and
        :type input_variables:
        :param export_variable_names:
        :type export_variable_names:
        :return: tuple of (1) return value of formula, (2) set of result variables, (3) the DSL object
        :rtype:
        """

        value_gen_partial = partial(
            lambda val: sample_value_for_simulation(float(val), sim_control, "inline constant"))

        d = dsl.EvalVisitor(variables=DSL_variable_dict, value_generator=value_gen_partial)

        for formula_name, variable in input_variables.items():
            logger.debug(f"sourcing variable {formula_name}")
            value = variable.data_source.get_value(formula_name, simulation_control=sim_control,
                                                   **{'process_name': kwargs.get('process_name', None)})
            # assert value.m.index.names == ['time', 'samples']
            # logger.debug(f'got value {value} with unit {value.units}')
            self.variable_cache[formula_name] = value
            d.variables[formula_name] = value

        d = dsl.evaluate(self.formula.text, visitor=d)

        export_variables = {}
        if export_variable_names:
            for model_var_name, global_var_name in export_variable_names.items():
                q_variable = d.variables[model_var_name]
                # logger.debug(model_var_name)
                # assert q_variable.m.index.names == ['time', 'samples']
                export_variables[global_var_name] = Variable.static_variable(global_var_name,
                                                                             q_variable)

        return export_variables, d

    def to_json(self) -> Dict[str, Any]:
        return self.formula.text


class FormulaProcess(object):

    def __init__(self, name: str, formulaModel: FormulaModel, input_variables: Dict[str, Variable] = None,
                 export_variable_names: Dict[str, str] = None, import_variable_names: Dict[str, Any] = None,
                 DSL_variable_dict=None, metadata=None):
        """

        :param name:
        :type name:
        :param formulaModel:
        :type formulaModel:
        :param export_variable_names: map describing what formula variables should be made available outside of the formula and under what global name
        :type import_variable_names: list of variable names that should be imported at evaluation time
        """
        self.name = name
        self.formulaModel = formulaModel
        # Req FP 0. FormulaProcesses can receive direct input variables at initiation time
        # Req FP 3. direct input and result variables are stored with the process models
        self.input_variables = input_variables
        self.export_variable_names = export_variable_names
        # Req FP 1. FormulaProcesses can also define import variables that can come from a pool of global variables
        self.import_variable_names = import_variable_names
        self._DSL_variables = DSL_variable_dict if DSL_variable_dict else {}

        self.aggregation_functions = defaultdict(lambda: sum)
        self.metadata = metadata if metadata is not None else {}

    def evaluate(self, sim_control, ingress_variables: Dict[str, Variable], debug=False) -> Dict[str, Variable]:
        variables = {}
        if ingress_variables:
            variables.update(ingress_variables)
        if self.input_variables:
            variables.update(self.input_variables)

        logger.debug(f"evaluating forumla in model {self.name}")
        export_variables, _DSL = self.formulaModel.evaluate(sim_control,
                                                            input_variables=variables,
                                                            export_variable_names=self.export_variable_names,
                                                            DSL_variable_dict=self._DSL_variables,
                                                            **{'process_name': self.name}
                                                            )

        self._DSL_variables = _DSL.variables
        self._DSL = _DSL
        return export_variables

    def import_variables_from_incoming_edge(self, incoming_edge) -> Dict[str, Tuple['ProcessModel', Variable]]:
        """
        Used by a process node to import variables from the upstream process adjacent to the incoming edge.

        Only variables included in the process :attribute: `ProcessModel.import_variables_list()` will be imported.

        Variables that were explicitly defined with the process will not be overridden.

        :param incoming_edge:
        :return: a dictionary of variable variable_name keys with tuples of the process name the variable came from and the variable value
        """

        imported_variables = {}
        for variable_name, variable in incoming_edge[2].items():
            # Req FP 2. During variable import, input variables (those defined at process initialisation) are not overwritten by import variables
            if variable_name in self.input_variables.keys():
                logger.warning(
                    f'Node {self.name} already has '
                    f'attribute {variable_name}. Not importing from incoming edge from node {incoming_edge[0]}')
            else:
                if isinstance(variable, Variable) and variable_name in self.import_variable_names:
                    logger.debug(
                        f'Node {self.name}: Adding variable {variable_name} from '
                        f'incoming edge {incoming_edge[0]} during import.')
                    adjacent_process = incoming_edge[0]
                    imported_variables[variable_name] = (adjacent_process, variable)
        return imported_variables

    def __str__(self):
        return f'{self.name}'

    def to_json(self) -> Dict[str, Any]:
        # @todo incomplete
        return {'formula': self.formulaModel.to_json()}

    @classmethod
    def from_json(cls, cm_json):
        # variables: List[Variable] = Variable.from_json_list(cm_json['variables'])
        variables = list(map(Variable.from_json, cm_json["variables"]))
        # instantiate by name from type key with component name and variables
        klass = klass_from_obj_type(cm_json)
        # instantiate with the variable instance field items
        return klass(cm_json['name'], variables)


class ServiceModel(object):
    """Calculate the energy_footprint of the service"""
    process_graph: DiGraph

    def __init__(self, name: str = None, process_graph: DiGraph = None):

        self.name = name
        self.process_graph: DiGraph = process_graph if process_graph else DiGraph()

    def link_processes(self, *args):
        """
        Add links to pairs of nodes from args
        :param args: a list of nodes
        :return:
        """
        a = args
        for i in range(len(a) - 1):
            self.process_graph.add_edge(a[i], a[i + 1])

    def add_process(self, configuration_node):
        logger.debug(f"Adding configuration {configuration_node.name} to model {self.name}")
        self.process_graph.add_node(configuration_node)

    def remove_process(self, process_name, include_children=True):
        # find nodes by name
        nodes = [node for node in self.process_graph.nodes_iter() if node.name == process_name]

        # add child nodes to list of to-be-removed nodes
        if include_children:
            subtree = ancestors(self.process_graph, nodes[0])
            # print(subtree)
            nodes = subtree.union(nodes)

        # do the removal
        for node in nodes:
            logger.info(f"removing node {node.name} from graph")
            self.process_graph.remove_node(node)

    def footprint(self, simulation_control=None, embodied=True, **kwargs) -> Dict[str, Dict[str, float]]:

        assert simulation_control is not None

        logger.info(f"calculating footprint for model {self.name}")
        G = self.process_graph

        unprocessed: List[FormulaProcess] = [x for x in G.nodes_iter() if G.in_degree(x) == 0]
        processed: List[FormulaProcess] = []

        results = {}

        """
        elements:[
            {group: 'nodes', data: { id: x}},
            {group: "edges", data: { id: x,  source: 'a', target: 'b' }},

        ]
        { nodes:{
            'Laptop':
            {
                'input vars': {'power':51},
                'import vars': [
                {    'formula_name': 'test',
                    'edge_name': 'power',
                    'value': 15}
                ]

            }
            }
        edges:[
            {'sourceNode':'', 'targetNode': '',
                data: {
                    'time': val
                }
            }
        ]

        }
        """
        simulation_control.trace = []

        while unprocessed:
            logger.debug(f"Next iteration of processing. ")
            logger.debug(f"Unprocessed queue: {[n.name for n in unprocessed]}")

            process_node = unprocessed.pop(0)

            if process_node not in processed:
                next_up = [process_node]
                while next_up:
                    process_node = next_up.pop()

                    node_trace_obj = {"group": "nodes", "data": {'id': process_node.name}}

                    logger.debug(f"Taking next node from next up queue. New state: {[n.name for n in next_up]}")

                    # any in-edges not processed? i.e. other subtrees incomplete
                    if any([edge for edge in self.process_graph.in_edges(process_node, data=True) if
                            'processed' not in edge[2]]):
                        logger.debug(f"Node has unprocessed children. Adding to back of unprocessed queue")
                        # --> push back to end of queue
                        unprocessed.append(process_node)
                        logger.debug(f"Unprocessed queue: {[n.name for n in unprocessed]}")
                        continue

                    logger.debug(f"Calculating process {process_node}")

                    # import incoming flows from downstream nodes
                    import_variables = defaultdict(list)
                    for edge in self.process_graph.in_edges(process_node, data=True):
                        logger.debug(
                            f"Importing flows from downstream processes on edge to <{process_node.name}>.")
                        edge_variables = process_node.import_variables_from_incoming_edge(edge)

                        logger.debug(f"Edge variables: {edge_variables}.")

                        for variable_name, process_variable_tuple in edge_variables.items():
                            import_variables[variable_name].append(process_variable_tuple)

                    def create_aggregate_variable_from_variables(name, process_var_list):
                        var_values = [
                            simulation_control.get_variable_value(process_variable_tuple[1], process_variable_tuple[0])
                            for process_variable_tuple in process_var_list]
                        res = process_node.aggregation_functions[name](var_values)
                        v = Variable.static_variable(name, res)  # b/s * s = b

                        return v

                    # Req FP 4. imported variables from adjancent downstream processes in the graph can be aggregated

                    import_node_trace_data = []

                    aggregated_import_vars = {}
                    for name, var_list in import_variables.items():
                        if process_node.import_variable_names[name]['aggregate']:
                            incoming_name = process_node.import_variable_names[name].get('formula_name', name)
                            aggregated_import_vars[incoming_name] = create_aggregate_variable_from_variables(name,
                                                                                                             var_list)
                            # value = aggregated_import_vars[incoming_name].data_source.get_value(None,
                            #                                                                     simulation_control)
                            # value_mean = value.pint.m.mean()
                            # todo - do we need trace data?
                            # import_node_trace_data.append({'formula_name': incoming_name,
                            #                                'edge_name': name,
                            #                                'value': value_mean,
                            #                                'unit': str(value_mean.units)})

                    ingress_variables = aggregated_import_vars

                    node_trace_obj['data']['import_vars'] = import_node_trace_data

                    export_variables = process_node.evaluate(simulation_control, ingress_variables,
                                                             debug=kwargs.get('debug', False))

                    # todo - is this tracing stuff still needed?
                    # _, input_vars = collect_process_variables(process_node)
                    # node_trace_obj['data']['input_vars'] = {
                    #     k: {'value': v, 'unit': str(v.pint.units)} for k, v in input_vars.items()}
                    # simulation_control.trace.append(node_trace_obj)

                    # @todo this does not work - result is never None as the DSL parser does not support return values
                    # @todo https://bitbucket.org/dschien/ngmodel_generic/issues/10/parser-allow-return-values
                    result = None
                    if result is not None:
                        # assert result.m.index.names == ['time', 'samples']
                        results[process_node.name] = result

                    # propagate flows to upstream nodes
                    for edge in self.process_graph.out_edges(process_node, data=True):
                        logger.debug(f"Propagating flows to upstream processes on edge to <{edge[1].name}>.")
                        # process_node.export_variables(edge, simulation_control)
                        edge[2].update(export_variables)

                        # edge_trace = {"group": "edges",
                        #               'data': {k: {'value': v.data_source.get_value(None, simulation_control).mean(),
                        #                            'units': str(
                        #                                v.data_source.get_value(None, simulation_control).pint.units)}
                        #                        for k, v in
                        #                        export_variables.items()},
                        #               }
                        # edge_trace['data'].update(
                        #     {'source': edge[0].name, 'target': edge[1].name, 'id': f'{edge[0].name}<>{edge[1].name}'})
                        #
                        # simulation_control.trace.append(edge_trace)

                        logger.debug(f"Marking out edge as processed (to node {edge[1].name}.)")
                        # mark this process_node as processed by adding attribute to out edges
                        edge[2]['processed'] = True

                        logger.debug(f"Adding node {edge[1].name} to next_up.")
                        next_up.append(edge[1])
                        logger.debug(f"Next_up queue state: {[n.name for n in next_up]}")
                    processed.append(process_node)

        # collect results
        return {'use_phase_energy': results}

    def collect_calculation_traces(self) -> Dict[str, pd.DataFrame]:
        """

        :return: dict of variable names (e.g duration, power, etc), with dataframes of process data

        """
        traces = defaultdict(dict)

        for process, data in self.process_graph.nodes(data=True):
            logger.debug(f'collecting calculation trace for process {process.name}')

            for k, v in process._DSL_variables.items():
                # if 'duration' == k:
                traces[k][process.name] = v

        return traces

    def collect_input_variables(self) -> Dict[str, Dict[str, float]]:

        variables = defaultdict(dict)

        for process in self.process_graph.nodes():
            variables[process.name] = process._DSL.variables

        return variables

    def to_json(self) -> Dict[str, Any]:
        processes: List[FormulaProcess] = self.process_graph.nodes()

        def make_edge_json(edge: Tuple[FormulaProcess, FormulaProcess]) -> Dict[str, str]:
            """
            :param edge: a tuple containing the source and target processes.
            :return: an edge converted from using process indices to process names.
            """
            return {
                "sourceName": edge[0].name,
                "targetName": edge[1].name

            }

        processes_json = list(map(lambda process: process.to_json(), processes))
        edges_json = list(map(make_edge_json, self.process_graph.edges()))

        return {
            "processes": processes_json,
            "edges": edges_json
        }

        # if isinstance(self, ServiceModel):
        #     return {'process_graph': json_graph.node_link_data(self.process_graph), 'name': self.name}

    @classmethod
    def from_json(cls, service_model_json):

        # parse the graph
        G = service_model_json['process_graph']

        new_nodes = []
        for node in G['nodes']:
            # print(node)
            new_node = FormulaProcess.from_json(node['id'])
            new_nodes.append({'id': new_node})
        # new_nodes = Variable.from_json_list([list(ob.values())[0] for ob in m2['nodes']])
        G['nodes'] = new_nodes
        # print(m2)

        process_graph = json_graph.node_link_graph(G, directed=True, multigraph=False)

        return ServiceModel(process_graph=process_graph, name=service_model_json['name'])


def complex_encoder(obj):
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    if type(obj).__module__ == np.__name__:
        return obj.tolist()
    raise TypeError(repr(obj) + " is not JSON serializable")


def complex_decoder(obj):
    if 'process_graph' in obj:
        return ServiceModel.from_json(obj)

    return obj
    # return ServiceModel.from_json(obj)


def set_child_subtree_metadata(cdn, model, value, category=None):
    subtree = nx.ancestors(model.process_graph, cdn)
    # print(subtree)
    nodes = subtree.union([cdn])
    for node in nodes:
        node.metadata[category] = value
        # setattr(node, category, value)


def generate_static_variable(sim_control, name=None, value=None, random=False) -> Variable:
    """
    Generates a static variable.

    If the sim control is set to time series = false, then a static value is replicated
    if the sim control is set to time series = true, then sampling from a normal dist is performed

    :param sim_control:
    :param name:
    :param value:
    :return:
    """

    value = sample_value_for_simulation(value, sim_control, name, random=random)
    return Variable.static_variable(name, value)


def sample_value_for_simulation(value, sim_control, name, random=False, unit=None):
    t = 1 if not sim_control.use_time_series else len(sim_control.times)
    size = t * sim_control.sample_size
    if random:
        value = np.random.normal(value, value / 10, size=size)
    else:
        value = np.full(size, value)
    if sim_control.use_time_series:
        value = gen_static_multi_index_df(None, name, sim_control.__dict__, distribution=value, unit=unit)
    else:
        value = np.full(size, value)
    return value


def collect_process_variables(process_node) -> Tuple[Dict[str, Variable], Dict[str, Variable]]:
    input_vars = {}
    import_vars = {}
    for var_name, var_value in process_node._DSL_variables.items():
        logger.debug(f'Collecting variable {var_name} from process {process_node.name}')
        # assert var_value.m.index.names == ['time', 'samples']
        import_var_names = []
        for _iv in process_node.import_variable_names.keys():
            _import_var_name = process_node.import_variable_names[_iv].get('formula_name', _iv)
            import_var_names.append(_import_var_name)

        if var_name in import_var_names:
            import_vars[var_name] = var_value
        if var_name in process_node.input_variables.keys():
            input_vars[var_name] = var_value
    return import_vars, input_vars

# def recreate_multiindex_df(q_ndarray, sim_control: SimulationControl):
#     df = pd.DataFrame(q_ndarray.m)
#     df.set_index(sim_control._df_multi_index, inplace=True)
#     q_ndarray._magnitude = df
