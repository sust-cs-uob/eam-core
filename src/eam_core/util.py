import copy
import time
from functools import partial

import matplotlib
import pint
import os
from pint.quantity import _Quantity

from eam_core.YamlLoader import YamlLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eam_core import Q_, FormulaProcess, collect_process_variables, SimulationControl
import csv
import inspect
import logging

import subprocess
from operator import itemgetter
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pypandoc
import simplejson
from networkx.drawing.nx_pydot import to_pydot
from tabulate import tabulate

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot

import errno

logger = logging.getLogger(__name__)


def find_node_by_name(model, name) -> FormulaProcess:
    return [node for node in model.process_graph.nodes() if node.name == name][0]


def store_trace_data(model, trace_data: Dict[str, Dict[str, pd.Series]], simulation_control=None, average=True):
    """

    :param simulation_control:
    :type simulation_control:
    :param trace_data: takes traces from :function:`ngmodel.collect_calculation_traces`
    :return:
    """
    if not os.path.exists(f'{simulation_control.output_directory}/traces/'):
        os.makedirs(f'{simulation_control.output_directory}/traces/')

    metadata = {}
    for var, df in trace_data.items():
        if isinstance(df, Q_):
            metadata[var] = {'unit': df.units}
            df = df.m
        if average:
            logger.warning(f'takign average of {var}')
            if isinstance(df.index, pd.core.index.MultiIndex):
                df = df.mean(level='time')

        # h5store(f'{simulation_control.output_directory}/traces/{var}.h5', df, **metadata)
        df.to_pickle(f'{simulation_control.output_directory}/traces/{var}.pdpkl')


def load_trace_data(output_directory, variable: str, base_dir='.') -> pd.DataFrame:
    return pd.read_pickle(f'{base_dir}/{output_directory}/traces/{variable}.pdpkl')


def store_calculation_debug_info(model, simulation_control, store_input_vars=True, average=True, target_units=None,
                                 result_variables=None):
    """

    Store

    :param model:
    :param simulation_control:
    :param store_input_vars:
    :return:
    """
    # store calculation traces
    traces = model.collect_calculation_traces()

    store_trace_data(model, traces, simulation_control=simulation_control, average=average)

    if store_input_vars:
        "calculate sensitivity for all variables"
        all_vars = model.collect_input_variables()

        flattened_vars = {}
        with open(f'{simulation_control.output_directory}/input_vars.csv', 'w') as f:
            writer = csv.writer(f)

            for proc_name, vars in sorted(all_vars.items()):
                for var_name, var in sorted(vars.items()):
                    val = to_target_dimension(var_name, var.mean(), target_units)
                    mean = val
                    writer.writerow((proc_name, var_name, mean.m, str(mean.units)))
                    flattened_vars[f'{proc_name}.{var_name}'] = mean

        df = pd.DataFrame.from_dict(flattened_vars, orient='index')
        df = df

        if not os.path.exists(f'{simulation_control.output_directory}/pd_pickles/'):
            os.makedirs(f'{simulation_control.output_directory}/pd_pickles/')

        h5store(f'{simulation_control.output_directory}/pd_pickles/input_variables.hd5', df, **metadata)
        df.to_pickle(f'{simulation_control.output_directory}/pd_pickles/input_variables.pdpkl')


def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('model_data', df)
    store.get_storer('model_data').attrs.metadata = kwargs
    store.close()


def h5load(filename):
    store = pd.HDFStore(filename)
    data = store['model_data']
    metadata = store.get_storer('model_data').attrs.metadata
    store.close()
    return data, metadata


def store_dataframe(q_dict: Dict[str, pd.Series], simulation_control=None, target_units=None, variable_name=None,
                    subdirectory=''):
    storage_df, metadata = pandas_series_dict_to_dataframe(q_dict, target_units=target_units, var_name=variable_name,
                                                           simulation_control=simulation_control)
    logger.info(f'metadata is {metadata}')
    filename = f'{simulation_control.output_directory}/{subdirectory}/result_data_{variable_name}.hdf5'

    if not os.path.exists(f'{simulation_control.output_directory}/{subdirectory}'):
        os.mkdir(f'{simulation_control.output_directory}/{subdirectory}')

    h5store(filename, storage_df.pint.dequantify(), **metadata)


def load_as_qantity_dict(filename) -> Dict[str, _Quantity]:
    val, metadata = h5load(filename)
    result_dict = {k: Q_(val[k], metadata[k]['unit']) for k in val.columns}
    return result_dict


def load_as_df_qantity(filename) -> _Quantity:
    val, metadata = h5load(filename)
    # check all units are the same
    # logger.debug(metadata.items())
    # assert len({v['unit'] for _, v in metadata.items()}) == 1
    return val.pint.quantify(level=-1), metadata
    # return Q_(val, list(metadata.values())[0]['unit'])


def load_as_plain_df(filename):
    data, metadata = load_as_df_qantity(filename)
    units = {v[0]: v[1] for v in data.pint.dequantify().columns.values}
    df = data.pint.dequantify()
    df.columns = df.columns.droplevel(1)
    return df, units


def load_as_df(filename) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    val, metadata = h5load(filename)

    return val, metadata


def pandas_series_dict_to_dataframe(data: Dict[str, pd.Series], target_units=None, var_name=None,
                                    simulation_control: SimulationControl = None):
    """
    input dict of process keys with result pd.Series

    output pd.DataFrame with with result pd.Series for columns of process keys

    """
    metadata = {}
    if simulation_control.with_group:
        results_df = pd.DataFrame(index=simulation_control.group_df_multi_index)
    else:
        results_df = pd.DataFrame(index=simulation_control._df_multi_index)
    for process, variable in data.items():
        logger.debug(f'converting results for process {process}')
        if target_units:
            variable = to_target_dimension(var_name, variable, target_units)
        results_df[process] = variable
        metadata[process] = variable.pint.units
    return results_df, metadata


def quantity_dict_to_dataframe(q_data: Dict[str, _Quantity], target_units=None, var_name=None,
                               simulation_control: SimulationControl = None) \
        -> Tuple[pd.DataFrame, Dict[str, str]]:
    data = None
    metadata = {}
    logger.debug(f'result data has the following processes {q_data.keys()}')
    for process, results in q_data.items():
        logger.debug(f'Converting <Quantity> back to <Pandas DF> {process}')
        if isinstance(results, Q_):

            if target_units:
                results = to_target_dimension(var_name, results, target_units)

            d = results.m

            metadata[process] = {'unit': str(results.units)}

            results_df = pd.DataFrame(data=d)
            results_df.columns = [process]
            if not isinstance(results_df.index, pd.MultiIndex):
                results_df.index = simulation_control._df_multi_index

        if data is None:
            data = results_df
        else:
            data[process] = results_df
    # print(data)
    return data, metadata


def generate_model_definition_markdown(model, in_docker=False, output_directory=None):
    """
    Generate markdown file with model code.
    :param model:
    :return:
    """
    H = nx.relabel_nodes(model.process_graph, lambda n: n.name)
    # write_dot(H, 'models/graphs/baseline.dot')
    pydot = to_pydot(H)

    input_vars = model.collect_input_variables()
    traces = model.collect_calculation_traces()

    df = load_df(model.name, 'all', output_directory=output_directory)

    logger.debug("Building model documentation markdown")

    with open(f'{output_directory}/{model.name}_model_documentation.md', 'w') as f:
        f.write("# Model\n\n")

        for node in pydot.get_node_list():
            name = node.get_name().strip('"')
            process_node = find_node_by_name(model, name)

            # method = process_node.formulaModel.formula
            # write_formula(f, method, name, 'energy_footprint')
            #
            # method = process_node.embodied_carbon_footprint
            # write_method(f, method, name, 'embodied_carbon_footprint')
            #
            # if hasattr(process_node, 'get_allocation_coefficient'):
            #     method = process_node.get_allocation_coefficient
            #     write_method(f, method, name, 'get_allocation_coefficient')

            items = []

            logger.debug(f"Processing process {name}")
            for item in input_vars[name].items():
                logger.debug(f"Processing variable {item[0]}")
                dataa = item[1]
                if isinstance(dataa, Q_):
                    dataa = dataa.m
                    metadata[item[0]] = {'unit': dataa.units}
                if dataa.index.nlevels == 2:
                    item__mean = dataa.mean(level='time').mean()
                else:
                    item__mean = dataa.mean()
                items.append((item[0], item__mean))

            # collect all the traces of this process
            process_traces = {key: traces[key][name] for key in traces.keys() if name in traces[key]}

            for item in process_traces.items():
                if item[1].index.nlevels == 2:
                    item__mean = item[1].mean(level='time').mean()
                else:
                    item__mean = item[1].mean()
                items.append((item[0], item__mean))

            f.writelines(tabulate(items, ['variable', 'value'], tablefmt='simple'))

            f.write('\n\n')

            items = []
            for metric, unit in zip(['use_phase_energy', 'use_phase_carbon', 'embodied_carbon'],
                                    ['J', 'gCO2e', 'gCO2e']):
                if name + '_' + metric in df.columns:
                    v = df[name + '_' + metric].mean()
                    items.append([metric, v, unit])

            f.writelines(tabulate(items, ['variable', 'value', 'unit'], tablefmt='simple'))

            f.write('\n\n')
    logger.info("writing pandoc")
    # if in_docker:
    #     ps = subprocess.Popen((
    #         f"docker run -v `pwd`:/source jagregory/pandoc -f markdown -t latex {simulation_control.output_directory}/{model.name}_model_documentation.md -o {simulation_control.output_directory}/{model.name}_model_documentation.pdf -V geometry:margin=0.2in, landscape"),
    #         shell=True)
    # else:
    logger.info(f'converting model doc at {output_directory}/{model.name}_model_documentation.md')
    output = pypandoc.convert_file(f'{output_directory}/{model.name}_model_documentation.md', 'pdf',
                                   outputfile=f'{output_directory}/{model.name}_model_documentation.pdf',
                                   extra_args=['-V', 'geometry:margin=0.2in, landscape'])


def write_method(f, method, name, suffix=None):
    f.writelines(["## Process {} - {}\n".format(name, suffix), "\n", "```python\n"])
    f.writelines(inspect.getsource(method))
    f.writelines(["```\n", "\n"])


def get_unit(name, process_node=None):
    unit = None
    if process_node:
        # try resolving the unit from the process documentation
        logger.debug(f'searching for unit for var {name} in process {process_node.name}')
        try:
            units = [info.unit for info in process_node.variable_infos() if info.property_name == name]
        except:
            units = None

        unit = units[0] if units else ''
    if not unit:
        # if that fails, use these defaults
        unit_dict = {'data_volume': 'b', 'use_phase_energy': 'J', 'use_phase_carbon': 'tCO2e', 'on_energy': 'J',
                     'active_standby_energy': 'J', 'viewing_time_minutes_monthly': 'minutes/month'}
        if name in unit_dict.keys():
            return unit_dict[name]

    return unit


def convert_for_graph(val, unit):
    if unit == 'b':
        return val * 1.25e-16, 'PB'
    if unit == 'J':
        return val * 2.77778e-13, 'GWh'
    if unit == 'hours/month':
        return val * 3.61984E-12, 'viewer-years/sec'
    if unit == 'minutes/month':
        return val * 2.60628E-09, 'viewer-years/h'

    return val, unit


def get_unit_by_name(name):
    if name == 'data_volume':
        return 'b'


def draw_graph_from_dotfile(model, file_type='pdf', show_variables=True, metric=None, start_date=None, end_date=None,
                            colour_def=None, show_histograms=True, in_docker=False, output_directory=None,
                            edge_labels=False, target_units=None):
    if show_histograms:
        generate_graph_node_barcharts(model.name, metric, start_date=start_date, end_date=end_date,
                                      base_directory=output_directory)

    project_dir = os.getcwd()  # '/Users/csxds/workspaces/ngmodel'

    H = nx.relabel_nodes(model.process_graph, lambda n: n.name)
    ref_period = 'monthly'
    H.graph['graph'] = {'label': f'{model.name} ({ref_period})', 'labelloc': 't', 'nodesep': 1, 'ranksep': 1}
    # write_dot(H, 'models/graphs/baseline.dot')
    pydot = to_pydot(H)

    for node in pydot.get_node_list():
        node.set_shape('box')
        node_name = node.get_name().strip('"')
        process_node = find_node_by_name(model, node_name)

        # attributes = '\n'.join(keys)
        logger.debug(f'drawing node {process_node.name}')
        node_colour = colour_def.get('colours', {}).get(
            process_node.metadata.get(colour_def.get('category_name', ""), None),
            '#EB0EAA')
        lable_txt = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD BGCOLOR="{node_colour}"></TD></TR>' \
                    + f'<TR><TD><FONT FACE="times bold italic">{node_name}</FONT></TD></TR>'

        if show_variables:

            import_vars, input_vars = collect_process_variables(process_node)

            lable_txt += '<TR><TD ALIGN="CENTER" BORDER="1">INPUT VARS</TD></TR>'
            for var_name, var_value in sorted(input_vars.items()):
                val = to_target_dimension(var_name, var_value, target_units)
                lable_txt = append_quantity_value_to_label(var_name, lable_txt,
                                                           val.pint.m.mean(), val.pint.units)
            lable_txt += '<TR><TD ALIGN="CENTER" BORDER="1">IMPORT VARS</TD></TR>'
            for var_name, var_value in sorted(import_vars.items()):
                val = to_target_dimension(var_name, var_value, target_units)
                lable_txt = append_quantity_value_to_label(var_name, lable_txt,
                                                           val.pint.m.mean(), val.pint.units)

            if process_node._DSL.result is not None:
                lable_txt += '<TR><TD ALIGN="CENTER" BORDER="1">RESULT</TD></TR>'
                val = process_node._DSL.result
                var_name = 'result'
                val = to_target_dimension(var_name, val, target_units)
                lable_txt = append_quantity_value_to_label(var_name, lable_txt,
                                                           val.pint.m.mean(), val.pint.units)

        img_file = process_node.name.replace(' ', '_')

        img_filename = f'{project_dir}/{output_directory}/subgraphs/{img_file}.png'

        if os.path.isfile(img_filename):
            lable_txt += f'<tr><td><IMG SCALE="TRUE" SRC="{img_filename}" /></td></tr>'
        lable_txt += '</TABLE>>'

        # lable_txt= "<<div><h1>{}</h1>{}</div>>".format(name, attributes)

        node.set_label(lable_txt)
        # node.set_tooltip("test")
        node.set_tooltip(process_node.formulaModel.formula.text)

    logger.info(f"Labelling edges to processes.")
    for dot_edge in pydot.get_edge_list():
        logger.debug(f"processing {dot_edge.obj_dict['points']}.")
        source_name = dot_edge.get_source()
        destination_name = dot_edge.get_destination()

        # @todo - why are some processes wrapped in double quotes?...
        def strip_quotes(string):
            if string.startswith('"') and string.endswith('"'):
                return string[1:-1]
            else:
                return string

        source_name = strip_quotes(source_name)
        destination_name = strip_quotes(destination_name)

        source_node = find_node_by_name(model, source_name)
        destination_node = find_node_by_name(model, destination_name)

        # import_variable_names = defaultdict(list)

        if edge_labels:
            logger.debug(f'processing edge: {source_name} -> {destination_name}')

            # find the edge object in
            for graph_edge in model.process_graph.in_edges(destination_node, data=True):
                # print(graph_edge)
                if graph_edge[0] == source_node:
                    # logger.debug(f'processing edge: {source_name} -> {destination_name}')
                    label = ""

                    for key, var in graph_edge[2].items():
                        if key == 'processed':
                            continue
                        # pass
                        logger.debug(f'adding label for variable {var.name}')
                        val = var.data_source.get_value(None, None, )
                        val = to_target_dimension(var.name, val, target_units)
                        # val_m = val.m.mean()
                        # unit = val.units
                        val_m = val.pint.m.mean()
                        val_unit = val.pint.units
                        if abs(val_m) > 0.01 and abs(val_m) < 100000:
                            label += '{} = {:.02f} {}\n'.format(var.name, val_m, val_unit)
                        else:
                            label += '{} = {:.2E} {}\n'.format(var.name, val_m, val_unit)

                        # dot_edge.set_label("this is a test ")
                        dot_edge.set_label(label)
                        # dot_edge.set_labeltooltip(label)

                    # for k, v in edge_variables.items():
                    #     import_variable_names[k].append(v)
    # @todo check model name does not allow code execution
    dot_file = f'{output_directory}/{model.name}.dot'
    pydot.write_dot(dot_file)

    cmd = r'perl -p -i.regexp_bak -e \'s/lp="\d+\.?\d*,\d*\.?\d*",\n?//\' "' + dot_file + '"'
    # l_cmd = ["perl", "-p", "-i.regexp_bak", "-e", '\'s/lp="\d+\.?\d*,\d*\.?\d*",\n?//\'', "{dot_file}"]

    import shlex
    l_cmd = shlex.split(cmd)

    logger.info(f'removing "lp" statements from {dot_file}')
    with subprocess.Popen(l_cmd, stdout=subprocess.PIPE) as proc:
        logger.info('output from shell process: ' + str(proc.stdout.read()))

    # time.sleep(2)

    dot_render_filename = f'{output_directory}/{model.name}_graph.{file_type}'
    logger.info('generating new graph plot from dot file at %s' % dot_render_filename)
    if in_docker:

        cwd = os.getcwd()
        cmd = f"docker run -v {cwd}:{project_dir} -w {project_dir} markfletcher/graphviz dot '{dot_file}' -T{file_type} -Gsplines=ortho -Grankdir=LR -Gnodesep=0.1 -Gratio=compress"

        logger.info(f'running docker cmd {cmd}')
        l_cmd = shlex.split(cmd)
        logger.info(f'running docker cmd {l_cmd}')
        with open(dot_render_filename, 'w') as output:
            with subprocess.Popen(l_cmd, stdout=output) as proc:
                pass
    else:
        cmd = f"dot '{dot_file}' -T{file_type} -Gsplines=ortho -Grankdir=BT > '{dot_render_filename}'"
        logger.info(f'running local cmd {cmd}')
        ps = subprocess.Popen((cmd), shell=True)


def to_target_dimension(name: str, q, target_units: Dict[str, str]):
    for target in target_units:
        func_name = list({v for v in target.keys() if v != 'to_unit'})[0]
        method_to_call = getattr(name, func_name)
        # @todo this is a security risk...
        result = method_to_call(target[func_name])
        if result:
            logger.debug(f"converting {name} to {target['to_unit']}")
            if isinstance(q, Q_):
                return q.to(target['to_unit'])
            if isinstance(q, pd.Series):
                return q.pint.to(target['to_unit'])

    return q


def append_quantity_value_to_label(att, lable_txt, val, unit):
    if abs(val) > 0.01 and abs(val) < 100000:
        lable_txt += '<TR><TD ALIGN="LEFT" >{} = {:.02f} {}</TD></TR>'.format(att, val, unit)
    else:
        lable_txt += '<TR><TD ALIGN="LEFT" >{} = {:.2E} {}</TD></TR>'.format(att, val, unit)

    return lable_txt


def get_unit_and_convert(att, process_node, val):
    if not isinstance(val, (float, int)):
        val = val.mean()
    if att in process_node.variables:
        var = process_node.variables[att]
    unit = get_unit(att, process_node)
    val, unit = convert_for_graph(val, unit)
    return unit, val


def generate_graph_node_barcharts(model_name, metric, start_date=None, end_date=None, base_directory=None):
    logger.info('generating graph node barcharts')

    filename = f'{base_directory}/result_data_{metric}.hdf5'

    df, units = load_as_plain_df(filename)

    if not start_date:
        start_date = df.index[0][0].date()
    if not end_date:
        end_date = df.index[-1][0].date()
    df = df.loc[start_date:end_date]

    s = [*df.mean(level='time').sum().items()]
    labels, values = zip(*sorted(s, key=itemgetter(1)))
    indexes = np.arange(len(labels))
    width = 1
    if not os.path.exists(f'{base_directory}/subgraphs/'):
        os.mkdir(f'{base_directory}/subgraphs/')

    for k, v in df.mean(level='time').sum().items():
        logger.debug(f'writing subgraph for {k}')
        fig, ax = plt.subplots(figsize=(3, 1))
        plt.bar(indexes, values, width, alpha=0.2)
        plt.bar(labels.index(k), values[labels.index(k)], width, color='red', alpha=0.7)
        plt.gca().annotate(k,
                           xy=(labels.index(k), values[labels.index(k)]), xycoords='data',
                           xytext=(-30, 20), textcoords='offset points',
                           arrowprops=dict(arrowstyle="->",
                                           connectionstyle="angle3,angleA=0,angleB=-90"))

        plt.xticks(indexes + width * 0.5, labels)

        plt.gca().get_xaxis().set_visible(False)

        plt.savefig(f'{base_directory}/subgraphs/%s.png' % k.replace(' ', '_'))
        plt.close()


def store_results(process_footprint_dict, model, simulation_control=None, ):
    if not os.path.exists(f'{simulation_control.output_directory}/pd_pickles/'):
        os.makedirs(f'{simulation_control.output_directory}/pd_pickles/')

    raw = {k: v.m for k, v in process_footprint_dict['use_phase_energy'].items()}

    dfm = pd.DataFrame.from_dict(raw)

    output_pickle_file = f'{simulation_control.output_directory}/pd_pickles/{model.name}_process_footprint_all.pdpkl'
    logger.info(f'saving process results to pickle {output_pickle_file}')
    dfm.to_pickle(output_pickle_file)

    # pdm = dfm.mean(level='time')

    # with open(f'{simulation_control.output_directory}/results.csv', 'w') as f:
    #
    #     w = csv.DictWriter(f, ['date'] + list(dfm.columns))
    #     w.writeheader()
    #
    #     for index_ts, row in dfm.to_dict(orient='index').items():
    #         row['date'] = index_ts.strftime("%Y/%m/%d")
    #         w.writerow(row)


def get_param_names(sim_control):
    arr = {}
    keys = sim_control.param_repo.parameter_sets.keys()
    # print(keys)
    for p_name in keys:
        # print(p_name)

        if sim_control.param_repo.exists(p_name):
            # ignore parameters that were not used as variables
            if sim_control.param_repo[p_name].cache is not None:
                arr[p_name] = sim_control.param_repo[p_name]
    return arr


def store_parameter_repo_variables(model, simulation_control=None):
    xlsxfile = f'{simulation_control.output_directory}/excel_variables_used.xlsx'
    names = get_param_names(simulation_control)
    data = {}
    for var_name in simulation_control.param_repo.parameter_sets.keys():
        if var_name in names:
            cs = {k: simulation_control.param_repo.parameter_sets[var_name]['default'].__dict__[k] for k in
                  ['comment', 'source']}
            data[var_name] = {**cs, **simulation_control.param_repo.parameter_sets[var_name]['default'].
                __dict__['kwargs']}

    pd.DataFrame(data).T.to_excel(xlsxfile)


def store_process_metadata(model, simulation_control=None):
    metadata = {}
    for n in model.process_graph:
        metadata[n.name] = n.metadata

    with open(f'{simulation_control.output_directory}/process_metadata.json', 'w') as f:
        f.write(simplejson.dumps(metadata))


kWh_p_J = 1 / 3.60E+06


def load_df(scenario, metric, group=None, mean_values=False, base_dir='.', output_directory=None, kind=None, **kwargs):
    if kind == 'input':
        df = pd.read_pickle(f'{base_dir}/{output_directory}/pd_pickles/input_variables.pdpkl')
    else:
        if group:
            df = pd.read_pickle(
                f'{base_dir}/{output_directory}/pd_pickles/{scenario}_process_footprint_{metric}_{group}.pdpkl')
        else:
            df = pd.read_pickle(
                f'{base_dir}/{output_directory}/pd_pickles/{scenario}_process_footprint_{metric}.pdpkl')
    if mean_values:
        return df.mean(level='time')
    return df


def store_sim_config(sim_control, directory, simulation_run_description, **kwargs):
    sim_control___dict__ = copy.deepcopy(sim_control.__dict__)
    del sim_control___dict__['times']
    del sim_control___dict__['param_repo']
    del sim_control___dict__['_df_multi_index']
    del sim_control___dict__['group_df_multi_index']
    sim_config = {'simulation_run_description': simulation_run_description, **sim_control___dict__}
    sim_config.update(kwargs)
    with open(f'{str(directory)}/sim_config.json', 'w') as outfile:
        simplejson.dump(sim_config, outfile, indent=4, sort_keys=True)


def create_output_folder(basedir):
    """

    :param basedir:
    :type basedir:
    :return: a directory in the form <output/model/timestamp/>
    :rtype:
    """
    timestr = time.strftime(Y_M_D_H_M_S)
    directory = basedir + timestr
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


Y_M_D_H_M_S = "%Y%m%d-%H%M%S"


def configue_sim_control_from_yaml(sim_control: SimulationControl, yaml_struct, output_directory):
    sim_control.sample_size = yaml_struct['Metadata'].get('sample_size', 1)
    sim_control.use_time_series = False

    if 'start_date' in yaml_struct['Metadata']:
        start_date = yaml_struct['Metadata']['start_date']
        end_date = yaml_struct['Metadata']['end_date']
        sim_control.use_time_series = True
        sim_control.times = pd.date_range(start_date, end_date, freq='MS')

    if 'sample_mean' in yaml_struct['Metadata']:
        sim_control.sample_mean_value = bool(yaml_struct['Metadata']['sample_mean'])

    if yaml_struct['Metadata'].get('table_file_name', '').endswith('csv'):
        sim_control.excel_handler = 'csv'

    if 'table_format_version' in yaml_struct['Metadata']:
        sim_control.table_version = yaml_struct['Metadata']['table_format_version']

    if yaml_struct['Metadata'].get('with_group', False):
        sim_control.with_group = True
        sim_control.groupings = yaml_struct['Metadata']['groupings']
        iterables = [sim_control.times, range(sim_control.sample_size), sim_control.groupings]
        index = sim_control.index_names.copy()
        index.append("group")
        sim_control.group_df_multi_index = pd.MultiIndex.from_product(iterables, names=index)
        sim_control.group_vars = yaml_struct['Metadata']['group_vars']
        sim_control.group_aggregation_vars = yaml_struct['Metadata'].get('group_aggregation_vars', None)

    sim_control.output_directory = output_directory
    iterables = [sim_control.times, range(sim_control.sample_size)]

    sim_control._df_multi_index = pd.MultiIndex.from_product(iterables, names=sim_control.index_names)


def prepare_simulation(model_output_directory, simulation_run_description, yaml_struct, scenario, sim_control=None,
                       filename=None,
                       IDs=False, formula_checks=False, **kwargs):
    if not sim_control:
        sim_control = SimulationControl()
        configue_sim_control_from_yaml(sim_control, yaml_struct, model_output_directory)
    sim_control.process_ids = IDs
    sim_control.model_run_datetime = time.strftime("%m%d-%H%M")
    sim_control.variable_ids = IDs
    sim_control.filename = filename
    sim_control.scenario = scenario
    args = kwargs.get('args', None)
    _kwargs = {}
    if args:
        _kwargs = vars(args)
        del kwargs['args']
    _kwargs.update(kwargs)
    store_sim_config(sim_control, model_output_directory, simulation_run_description, **_kwargs)
    create_model_func = partial(YamlLoader.create_service, yaml_struct, formula_checks=formula_checks)
    return create_model_func, sim_control, yaml_struct
