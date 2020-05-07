import os
import shlex
import subprocess
import sys
import time

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pip._vendor.pkg_resources import resource_filename, Requirement

import json as json

from shutil import copyfile
from eam_core.util import get_sim_run_description, create_output_folder, \
    draw_graph_from_dotfile, load_as_df_qantity, load_as_plain_df, prepare_simulation, find_node_by_name

from eam_core.YamlLoader import YamlLoader
from eam_core.common_graphical_analysis import load_metadata, plot_kind, plot_process_with_input_vars

from eam_core.dataframe_result_utils import group_data

from eam_core.SimulationRunner import SimulationRunner

import pandas as pd
from ruamel import yaml
import pint

from functools import partial

# from joblib import Parallel, delayed
# import multiprocessing
import itertools
import configparser

try:
    CONFIG_FILE = "local.cfg"
    cfg = configparser.ConfigParser()
    cfg.read_file(itertools.chain(['[global]'], open(CONFIG_FILE)))
    config = cfg['global']
except:
    config = {}

import logging
from logging.config import dictConfig

with open(resource_filename(Requirement.parse('eam_core'), "eam_core/logconf.yml"), 'r') as f:
    log_config = yaml.safe_load(f.read())
dictConfig(log_config)

logger = logging.getLogger(__name__)


def setup_parser(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_config', '-a', help="name of analysis_config to run")
    parser.add_argument('--comment', '-c',
                        help="Provide a comment to this simulation run. This will skip a UI prompt and can be used for headless environments.")
    parser.add_argument('--docker', '-d', help="generate graphviz graphs in a docker container", action='store_true')
    parser.add_argument('--filetype', '-f', help="generate graphviz graphs in a docker container", default='pdf')
    parser.add_argument('--local', '-l', help="don't check for updates cloud spreadsheets", action='store_true')
    parser.add_argument('--singlethread', '-s', help="don't use multicore speed ups", action='store_true')
    parser.add_argument('--verbose', '-v', help="enable debug level logging", action='store_true')
    parser.add_argument('yamlfile', help="yaml file to run")
    parser.add_argument('--sensitivity', '-n', help="run sensitivity analysis", action='store_true')
    parser.add_argument('--documentation', '-D', help="create documentation", action='store_false')
    parser.add_argument('--IDs', '-id', help="give each process and variable an ID", action='store_true')

    args = parser.parse_args(args)
    # print(args)
    return args


def write_formula(outstream, formula, name):
    outstream.write(f'## Process {name}\n\n')
    outstream.writelines(formula)
    outstream.write('\n\n')


def create_documentation(runner):
    """
    Generate markdown file with model code.
    :param model:
    :return:
    """
    from networkx.drawing.nx_pydot import to_pydot
    import networkx as nx
    import pypandoc
    model = runner.model
    output_directory = runner.sim_control.output_directory
    H = nx.relabel_nodes(model.process_graph, lambda n: n.name)
    # write_dot(H, 'models/graphs/baseline.dot')
    pydot = to_pydot(H)

    logger.debug("Building model documentation markdown")

    with open(f'{output_directory}/{model.name}_model_documentation.md', 'w') as f:
        f.write("# Model\n\n")

        for node in pydot.get_node_list():
            name = node.get_name().strip('"')
            process_node = find_node_by_name(model, name)

            method = process_node.formulaModel.formula.text
            write_formula(f, method, name)

            items = []

            logger.debug(f"Processing process {name}")
            # for item in input_vars[name].items():
            #     logger.debug(f"Processing variable {item[0]}")
            #     dataa = item[1]
            #     if isinstance(dataa, Q_):
            #         dataa = dataa.m
            #         metadata[item[0]] = {'unit': dataa.units}
            #     if dataa.index.nlevels == 2:
            #         item__mean = dataa.mean(level='time').mean()
            #     else:
            #         item__mean = dataa.mean()
            #     items.append((item[0], item__mean))
            #
            # # collect all the traces of this process
            # process_traces = {key: traces[key][name] for key in traces.keys() if name in traces[key]}
            #
            # for item in process_traces.items():
            #     if item[1].index.nlevels == 2:
            #         item__mean = item[1].mean(level='time').mean()
            #     else:
            #         item__mean = item[1].mean()
            #     items.append((item[0], item__mean))
            #
            # f.writelines(tabulate(items, ['variable', 'value'], tablefmt='simple'))

            f.write('\n\n')
            #
            # items = []
            # for metric, unit in zip(['use_phase_energy', 'use_phase_carbon', 'embodied_carbon'],
            #                         ['J', 'gCO2e', 'gCO2e']):
            #     if name + '_' + metric in df.columns:
            #         v = df[name + '_' + metric].mean()
            #         items.append([metric, v, unit])
            #
            # f.writelines(tabulate(items, ['variable', 'value', 'unit'], tablefmt='simple'))

            f.write('\n\n')

    if runner.use_docker:
        cwd = os.getcwd()
        # cmd = f"docker run -v {cwd}:{project_dir} -w {project_dir} markfletcher/graphviz dot {dot_file} -T{file_type} -Gsplines=ortho -Grankdir=LR -Gnodesep=0.1 -Gratio=compress"
        # cmd = f"docker run -v {cwd}:{project_dir} -w {project_dir} markfletcher/graphviz dot {dot_file} -T{file_type} -Gsplines=ortho -Grankdir=BT"
        # cmd = f"docker run -v {cwd}:{project_dir} -w {project_dir} markfletcher/graphviz dot {dot_file} -T{file_type} -Gsplines=ortho -Grankdir=BT > {dot_render_filename}"
        pdf_file_name = f"{runner.sim_control.output_directory}/{model.name}_model_documentation.pdf"
        cmd = f"docker run -v `pwd`:/source jagregory/pandoc -f markdown -t latex {runner.sim_control.output_directory}/{model.name}_model_documentation.md -o {pdf_file_name} -V geometry:margin=0.2in, landscape"
        logger.info(f'running docker cmd {cmd}')
        l_cmd = shlex.split(cmd)
        logger.info(f'running docker cmd {l_cmd}')
        # with open(pdf_file_name, 'w') as output:
        #     with subprocess.Popen(l_cmd, stdout=output) as proc:
        #         pass
        ps = subprocess.Popen(cmd, shell=True)
    else:
        output = pypandoc.convert_file(f'{output_directory}/{model.name}_model_documentation.md', 'pdf',
                                       outputfile=f'{output_directory}/{model.name}_model_documentation.pdf',
                                       extra_args=['-V', 'geometry:margin=0.2in, landscape'])


def run_scenario(scenario, model_run_base_directory=None, simulation_run_description=None, yaml_struct=None, args=None,
                 analysis_config=None, output_persistence_config=None):
    """
    Run a scenario
    :param scenario:
    :type scenario:
    :param model_run_base_directory:
    :type model_run_base_directory:
    :param simulation_run_description:
    :type simulation_run_description:
    :param yaml_struct:
    :type yaml_struct:
    :param args:
    :type args:
    :param analysis_config:
    :type analysis_config:
    :param output_persistence_config:
    :type output_persistence_config:
    :param mean_run:
    :type mean_run:
    :return:
    :rtype:
    """
    logger.info(f'Evaluating scenario {scenario}')
    # prepare a sub directory
    model_output_directory = model_run_base_directory + f"/{scenario}"
    if not os.path.exists(model_output_directory):
        os.makedirs(model_output_directory)

    mean_run = None

    if yaml_struct['Metadata']['sample_mean'] == False:
        # run with just mean values, so that we can deal with 'sub-zero' values during analysis
        # this is needed because of sampling effect when uncertainty is very high
        mean_run = run_mean(args, model_run_base_directory, simulation_run_description, yaml_struct, scenario)
    create_model_func, sim_control, yaml_struct = prepare_simulation(model_output_directory,
                                                                     simulation_run_description, yaml_struct,
                                                                     scenario, filename=args.yamlfile, IDs=args.IDs)
    if args.sensitivity:
        runner = SimulationRunner()
        model, variances = runner.run_SA(create_model_func=create_model_func, embodied=False, sim_control=sim_control)
        # model, variances = runner.run_OTA_SA(create_model_func=create_model_func, embodied=False, sim_control=None)

        df = pd.DataFrame(variances).T
        df.sort_values(by=['std_dev'], inplace=True)

        logger.debug(df)

        writer = pd.ExcelWriter(f'{model_output_directory}/sa.xlsx')
        df.to_excel(writer, 'OTA SA')
        writer.close()
        return (scenario, runner)

    runner = SimulationRunner()
    runner.use_docker = args.docker
    runner.sim_control = sim_control
    runner.run(create_model_func=create_model_func, debug=True,
               target_units=YamlLoader.get_target_units(yaml_struct),
               result_variables=yaml_struct['Analysis'].get('result_variables', []),
               output_persistence_config=output_persistence_config)

    analysis(runner, yaml_struct, analysis_config=analysis_config, mean_run=mean_run, image_filetype=args.filetype)
    # analysis model results

    if args.documentation:
        create_documentation(runner)

    return (scenario, runner)


def run_mean(args, model_run_base_directory=None, simulation_run_description=None, yaml_struct=None,
             scenario='default'):
    """

    :param args:
    :type args:
    :return:
    :rtype:
    """

    logger.info(f"running mean for scenario {scenario}")
    if not model_run_base_directory:
        model_run_base_directory, simulation_run_description, yaml_struct = load_configuration(args)

    import copy
    _yaml_struct = copy.deepcopy(yaml_struct)

    _yaml_struct['Metadata']['sample_mean'] = True
    _yaml_struct['Metadata']['sample_size'] = 1

    run_scenario_f = partial(run_scenario, model_run_base_directory=model_run_base_directory,
                             simulation_run_description=simulation_run_description, yaml_struct=_yaml_struct, args=args,
                             analysis_config={}, output_persistence_config={})

    _, runner = run_scenario_f(scenario)
    return runner


def run(args, analysis_config=None):
    """

    :param args:
    :type args:
    :param analysis_config: if None, the analysis_config in the args is read
    :type analysis_config:
    :return:
    :rtype:
    """
    # initialise this set of simulations runs by creating a directory for output files
    model_run_base_directory, simulation_run_description, yaml_struct = load_configuration(args)

    # setup any analysis configuration
    if analysis_config is None:
        if args.analysis_config:
            # find the analysis_config in the yaml file by name
            analysis_configs_ = [item for item in yaml_struct['Metadata'].get('analysis_configs', []) if
                                 item['name'] == args.analysis_config]
            if analysis_configs_:
                logger.info(f"Running with analysis_config '{analysis_configs_[0]}'")
                analysis_config = analysis_configs_[0]
            else:
                logger.warning(f"Could not find analysis_config {args.analysis_config}")
                analysis_config = {}

        else:
            analysis_config = {}

    # scenarios to evaluate
    scenarios = yaml_struct['Analysis'].get('scenarios', [])
    # 'default' is implicit
    scenarios.append('default')
    # define aspects/data to store
    output_persistence_config = {'store_traces': True}

    # a partial to invoke for each scenario
    run_scenario_f = partial(run_scenario, model_run_base_directory=model_run_base_directory,
                             simulation_run_description=simulation_run_description, yaml_struct=yaml_struct, args=args,
                             output_persistence_config=output_persistence_config, analysis_config=analysis_config)

    if True:
        results = []
        for scenario in scenarios:
            results.append(run_scenario_f(scenario))
    else:
        logger.info("Running in parallel")
        from pathos.multiprocessing import ThreadPool
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        results = ThreadPool(processes=num_cores).map(run_scenario_f, scenarios)
        # results = Parallel(n_jobs=num_cores)(delayed(run_scenario_f)(i) for i in scenarios)

    # combine all run results
    runners = dict((x, y) for x, y in results)

    analysis_config.update(yaml_struct['Analysis'])
    analysis_config.update(yaml_struct['Metadata'])

    scenario_paths = {scenario_name: run_data.sim_control.output_directory for scenario_name, run_data in
                      runners.items()}
    summary_analysis(scenario_paths, model_run_base_directory, analysis_config, yaml_struct,
                     image_filetype=args.filetype)
    return runners


def load_configuration(args):
    """
    model_run_base_directory

    :param args:
    :type args:
    :return:
    :rtype:
    """
    if not 'comment' in args.__dict__:
        simulation_run_description = get_sim_run_description()
    else:
        simulation_run_description = args.comment
    yamlfile = args.yamlfile
    yaml_struct = None
    with open(yamlfile, 'r') as stream:
        try:
            yaml_struct = yaml.load(stream, Loader=yaml.RoundTripLoader)
        except yaml.YAMLError as exc:
            logger.error(f'Error while loading yaml file {yamlfile} {exc}')
            sys.exit(1)
    model_basedir = f"output/{yaml_struct['Metadata']['model_name']}/"
    model_run_base_directory = create_output_folder(model_basedir)
    for file_location in yaml_struct['Metadata'].get('file_locations', []):
        if file_location['type'] == 'google_spreadsheet':
            google_id_alias = file_location['google_id_alias']

            sheet_id_alias_ = config['google_drive_params_sheet_id_alias']
            sheet_id_alias = json.loads(sheet_id_alias_)
            file_id = sheet_id_alias[google_id_alias]

            if not args.local:
                from eam_core.gdrive_data_helper import update_local_data
                logger.info(f'updating cloud file with alias "{google_id_alias}"')
                from eam_core.gdrive_data_helper import update_local_data
                modDTime, revision = update_local_data(file_id, file_location['file_name'])

            file_name = os.path.basename(file_location['file_name'])
            copyfile(file_location['file_name'], f'{model_run_base_directory}/{file_name}')

    else:
        logger.info("Running with local parameter data only.")
    return model_run_base_directory, simulation_run_description, yaml_struct


def plot_scenario_comparison(scenario_paths, model_run_base_directory, base_dir, yaml_struct, image_filetype=None):
    load_data = load_as_df_qantity
    variable = yaml_struct['Metadata']['comparison_variable']
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    numeric_values = {}
    max_ = 0
    for scenario_name, scenario_path in scenario_paths.items():
        output_directory = scenario_path
        # load dataframe of energy values
        pint_pandas_data, m = load_data(f'{output_directory}/result_data_{variable}.hdf5')
        units = {v[0]: v[1] for v in pint_pandas_data.pint.dequantify().columns.values}
        df = pint_pandas_data.pint.dequantify()
        df.columns = df.columns.droplevel(1)
        data = df
        base_dir = '.'
        metadata = load_metadata(output_directory, base_dir=base_dir)

        # if 'groups' in plot_def:
        #     data = group_data(data, metadata, plot_def)

        mean_ = data.mean(level='time').sum(axis=1)
        numeric_values[scenario_name] = mean_
        mean_.plot(ax=ax, kind='line',
                   # legend=False,
                   linewidth=1,
                   alpha=.3,
                   label=scenario_name
                   )
        mean__max = mean_.max()
        if mean__max > max_:
            max_ = mean__max

    ax.set_ylim(bottom=0, top=max_ * 1.1)
    ax.legend()

    plt.tight_layout()
    file_name = f'scenarios.{image_filetype}'
    if file_name:
        if not os.path.exists(f'{base_dir}/{model_run_base_directory}'):
            os.makedirs(f'{base_dir}/{model_run_base_directory}')
        file_name_ = f'{base_dir}/{model_run_base_directory}/{file_name}'
        logger.info(f'storing plot at {file_name_}')
        fig.savefig(file_name_)

    return numeric_values


def summary_analysis(scenario_paths, model_run_base_directory, analysis_config, yaml_struct, image_filetype=None):
    """
    Compare all scenarios with each other.

    """
    base_dir = '.'

    xlsx_file_name = f'{base_dir}/{model_run_base_directory}/summary_{time.strftime("%m%d-%H%M")}.xlsx'
    writer = pd.ExcelWriter(xlsx_file_name)
    sheet_descriptions = {}
    pd.DataFrame.from_dict(sheet_descriptions, orient='index').to_excel(writer, 'toc')

    numeric_values = plot_scenario_comparison(scenario_paths, model_run_base_directory, base_dir, yaml_struct,
                                              image_filetype=image_filetype)

    for scenario, numeric_vals in numeric_values.items():
        numeric_vals.to_excel(writer, f'mean {scenario} (comp)')

    # ======================== GO ================
    for variable in analysis_config.get('numerical', []):
        data = None
        unit = None

        logger.info(f'numerical analysis for var {variable}')
        for scenario_name, scenario_path in scenario_paths.items():
            scenario = scenario_name
            logger.info(f'numerical analysis for scenario {scenario}')

            scenario_directory = model_run_base_directory + f'/{scenario}'

            if data is None:
                data, units = load_as_plain_df(f'{scenario_directory}/result_data_{variable}.hdf5')

                # data.rename(lambda x: f'{x}.{scenario}', axis='columns', inplace=True)
                data = data.mean(level='time').mean().to_frame(name=scenario)
                # check all units are the same
                units_set = set(units[p] for p in units.keys())
                assert len(units_set) == 1

                unit = units_set.pop()


            else:
                _dframe, _ = load_as_plain_df(f'{scenario_directory}/result_data_{variable}.hdf5')
                # _dframe.rename(lambda x: f'{x}.{scenario}', axis='columns', inplace=True)

                # data = pd.concat([s1, s2], axis=1)
                data = data.join(_dframe.mean(level='time').mean().to_frame(name=scenario))
        sheet_name = f'mean {variable}'
        sheet_descriptions[
            sheet_name] = f'{sheet_name}: a direct load of the result data, monthly mean values. Unit: {unit}'

        logger.info("storing mean values to excel")

        data.to_excel(writer, sheet_name)
        # data.mean(level='time').mean().to_excel(writer, f'mean {variable} ')
        # writer.sheets[f'mean {variable}'].column_dimensions['A'].width = 30

    writer.save()

    logger.info("writing TOC")
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_file_name)
    ws = wb['toc']
    # ----------------------------------------------------- TV vs STB
    # ------------------ TOC ----------------
    for row, (name, desc) in enumerate(sheet_descriptions.items(), start=1):
        ws[f'A{row}'].value = name
        ws[f'B{row}'].value = desc
        ws[f'C{row}'].value = f'=HYPERLINK("#\'{name}\'!A1", "Link")'
    ws.column_dimensions["A"].width = '23'
    ws.column_dimensions["B"].width = '63'
    ws.column_dimensions["C"].width = '10'
    logger.debug(xlsx_file_name)
    wb.save(xlsx_file_name)


def analysis(runner, yaml_struct, analysis_config=None, mean_run=None, image_filetype=None):
    countries=True
    if analysis_config is None or not analysis_config:
        return
    model = runner.model
    sim_control = runner.sim_control
    scenario = sim_control.scenario

    # print(footprint_result_dict['use_phase_energy'].keys())
    # for k, v in footprint_result_dict['use_phase_energy'].items():
    #     print(k, v)
    # mean = SimulationRunner.get_stddev_and_mean(footprint_result_dict['use_phase_energy'], np.ones(2))
    # print(mean)

    if sim_control.use_time_series:
        output_directory = sim_control.output_directory

        logger.info('generate_model_definition_markdown')
        logger.warning('skipping markdown generation for now')
        # @todo fix generate_model_definition_markdown
        # generate_model_definition_markdown(model, in_docker=args.docker, output_directory=output_directory)

        if 'start_date' in yaml_struct['Metadata']:
            start_date = yaml_struct['Metadata']['start_date']
            end_date = yaml_struct['Metadata']['end_date']

        if scenario == 'default':

            if 'process_tree' in analysis_config:
                process_group_colour_defs = analysis_config['process_tree'].get('process_group_colours', {})
                show_variables = analysis_config['process_tree'].get('show_variables', True)

                draw_graph_from_dotfile(model, show_variables=show_variables,
                                        file_type=image_filetype, metric='energy',
                                        start_date=start_date, end_date=end_date, colour_def=process_group_colour_defs,
                                        in_docker=runner.use_docker,
                                        output_directory=output_directory,
                                        target_units=YamlLoader.get_target_units(yaml_struct), edge_labels=True,
                                        )
        logger.info('generating plots and tables')
        # generate_plots_and_tables(scenario=scenario, metric='use_phase_energy', base_dir='.', start_date=start_date,
        #                           end_date=end_date, output_directory=output_directory)
        base_dir = '.'
        metadata = load_metadata(output_directory, base_dir=base_dir)

        xlsx_file_name = f'{base_dir}/{output_directory}/results_{scenario}_{time.strftime("%m%d-%H%M")}.xlsx'
        writer = pd.ExcelWriter(xlsx_file_name)
        sheet_descriptions = {}
        pd.DataFrame.from_dict(sheet_descriptions, orient='index').to_excel(writer, 'toc')
        # df of x samples, monthly frequency between start and end date
        load_data = load_as_df_qantity

        # ======================== GO ================
        for variable in yaml_struct['Analysis'].get('numerical', []):
            logger.info(f'numerical analysis for var {variable}')

            pint_pandas_data, m = load_data(f'{output_directory}/result_data_{variable}.hdf5')
            units = {v[0]: v[1] for v in pint_pandas_data.pint.dequantify().columns.values}
            df = pint_pandas_data.pint.dequantify()
            df.columns = df.columns.droplevel(1)
            data = df
            unit = next(iter(units))

            sheet_name = f'mean {variable} '
            sheet_descriptions[
                sheet_name] = f'{sheet_name}: a direct load of the result data, monthly mean values. Unit: {unit}'
            logger.info("storing mean values to excel")
            if countries:
                mean = data.mean(level=['country','time']).mean(level='country')
            else: mean = data.mean(level='time').mean()
            mean.to_excel(writer, sheet_name)

            sheet_name = f'25 quantiles {variable}'
            sheet_descriptions[
                sheet_name] = f'{sheet_name}: a direct load of the result data, monthly mean values. Unit: {unit}'
            logger.info("storing quantile values to excel")
            if countries:
                low = data.abs().groupby(level=['country','time']).quantile(.25)
            else: low = data.abs().groupby(level=['time']).quantile(.25)
            low.to_excel(writer, sheet_name)

            sheet_name = f'75 quantiles {variable}'
            sheet_descriptions[
                sheet_name] = f'{sheet_name}: a direct load of the result data, monthly mean values. Unit: {unit}'
            if countries:
                high = data.abs().groupby(level=['country','time']).quantile(.75)
            else: high = data.abs().groupby(level=['time']).quantile(.75)
            high.to_excel(writer, sheet_name)

        # sum up monthly values to aggregate - duration depends on distance between start and end date
        #    load_data_aggegrate = lambda : load_data().sum(level='samples')
        xlabel = f'{unit}/a'
        common_args = {'start_date': start_date,
                       'end_date': end_date,
                       'base_dir': base_dir,
                       'xlabel': xlabel,
                       'metadata': metadata,
                       'output_scenario_directory': output_directory}

        logger.info("plot_platform_process_annual_total")

        plot_defs = yaml_struct['Analysis'].get('plots', [])

        for plot_def in plot_defs:
            name = plot_def['name']
            if name in analysis_config.get('named_plots', []):
                variable = plot_def['variable']
                logger.info(plot_def['name'])
                title = plot_def.get('title', 'Annual Total Energy Consumption')
                # sum up monthly values to aggregate - duration depends on distance between start and end date
                #    load_data_aggegrate = lambda : load_data().sum(level='samples')

                pint_pandas_data, m = load_data(f'{output_directory}/result_data_{variable}.hdf5')

                units = {v[0]: v[1] for v in pint_pandas_data.pint.dequantify().columns.values}
                df = pint_pandas_data.pint.dequantify()
                df.columns = df.columns.droplevel(1)
                data = df
                # check all units are the same
                units_set = set(units[p] for p in units.keys())
                assert len(units_set) == 1

                unit = units_set.pop()

                common_args.update({'unit': unit})

                if 'xlabel' in plot_def:
                    xlabel = plot_def['xlabel'].format(unit=unit)

                ylabel = None
                if 'ylabel' in plot_def:
                    ylabel = plot_def['ylabel'].format(unit=unit)

                common_args.update({'xlabel': xlabel, 'ylabel': ylabel})

                kind = plot_def.get('kind', 'box')
                print(kind)
                if kind == 'area':
                    # for area charts if mean_run parameter is not null, we can assume this analysis here is of a run
                    # with random sampling. In this case, we need the results from a 'non-sampling' run so zero-values
                    # don't mess up the statistics
                    mean_run_df = data
                    if mean_run:
                        mean_run_df, mean_run_metadata = mean_run.get_process_variable_values(variable,
                                                                                              YamlLoader.get_target_units(
                                                                                                  yaml_struct),
                                                                                              mean_run.sim_control)

                        mean_run_df = mean_run_df.pint.dequantify()
                        mean_run_df.columns = mean_run_df.columns.droplevel(1)
                if 'groups' in plot_def:
                    if mean_run:
                        mean_run_df = group_data(mean_run_df, metadata, plot_def)
                        common_args.update({'mean_data': mean_run_df})

                    data = group_data(data, metadata, plot_def)

                data.to_pickle(f'{output_directory}/graph_data_{name}.pdpkl')

                d = plot_kind(data, figsize=(15, 12), file_name=f'{scenario}_{name}.{image_filetype}', title=title,
                              kind=kind, **common_args)
            else:
                logger.warning(f"Named plot {plot_def['name']} not found")

        if 'individual_process_graphs' in analysis_config:
            # @todo - hack
            variable = yaml_struct['Metadata']['individual_process_graphs_variable']

            pint_pandas_data, m = load_data(f'{output_directory}/result_data_{variable}.hdf5')

            units = {v[0]: v[1] for v in pint_pandas_data.pint.dequantify().columns.values}
            # check all units are the same
            units_set = set(units[p] for p in units.keys())
            assert len(units_set) == 1

            unit = units_set.pop()

            df = pint_pandas_data.pint.dequantify()
            df.columns = df.columns.droplevel(1)
            data = df

            iv_dfs = plot_process_with_input_vars(model, sim_control, data, f'{scenario}_input_vars', output_directory,
                                                  unit, base_dir=base_dir, analysis_config=analysis_config,
                                                  image_filetype=image_filetype)

            # store in excel
            for proc, input_vars_df in iv_dfs.items():
                sheet_name = f'(iv) {proc}'[:30]
                sheet_descriptions[
                    sheet_name] = f'{sheet_name}: the input variables'
                input_vars_df.mean(level='time').to_excel(writer, sheet_name)

        if 'input_vars' in analysis_config:
            logger.info("plotting input vars")
            all_vars = model.collect_input_variables()

            _vars = {}

            iv_ac = None

            for _, p_vars in all_vars.items():
                _vars.update(p_vars)

            # remove all vars that were not defined in the analysis config
            input_vars_ac = analysis_config['input_vars']
            if input_vars_ac is not None and 'variables' in input_vars_ac:
                iv_ac = set(input_vars_ac['variables'])

                _vars = {k: v for k, v in _vars.items() if k in iv_ac}

            file_name = f'input_vars.{image_filetype}'

            import copy
            rcParams_bkp = copy.deepcopy(plt.rcParams)
            plt.rcParams['axes.grid'] = True
            plt.rcParams['axes.linewidth'] = .2
            plt.rcParams['axes.spines.top'] = True
            plt.rcParams['axes.spines.right'] = True
            plt.rcParams['axes.spines.bottom'] = True
            plt.rcParams['axes.spines.left'] = True
            plt.rcParams['font.size'] = 8
            plt.rcParams['grid.linestyle'] = ':'
            plt.rcParams['lines.linewidth'] = .1
            plt.rcParams['hatch.linewidth'] = 0.3

            import math
            n = len(_vars)
            # n = 50
            rows = math.ceil(n / 3)
            f, axarr = plt.subplots(rows, 3, sharex='col', sharey=False)
            f.set_size_inches(15, n * 0.5)

            var_names = sorted(_vars.keys())
            for i, ax in enumerate(axarr.flat):
                if i == len(var_names):
                    break
                var_name = var_names[i]
                val = _vars[var_name]
                logger.info(f"plotting input variable {var_name}")
                # for var_name, val in sorted(_vars.items()):
                #     fig = plt.figure(figsize=(10, 5))
                #     ax = fig.add_subplot(111)

                try:
                    data = _vars[var_name].pint.to_reduced_units()
                except:
                    logger.error(f"could not reduce units {_vars[var_name].pint.units}")
                    pass

                p_df = pd.DataFrame(data=data.pint.m)
                p_df.columns = [var_name]
                if not isinstance(p_df.index, pd.MultiIndex):
                    p_df.index = sim_control._df_multi_index

                mean_ = p_df.mean(level='time')
                mean_.plot(ax=ax, kind='line',
                           legend=False,
                           linewidth=1,
                           color='k', alpha=.3,
                           #                    marker="x"
                           )

                ax.set_ylabel(data.pint.units)
                #     fig.title(var_name)
                ax.set_title(var_name[:70], fontsize=7, loc='right')
                ax.yaxis.get_offset_text().set_size(6)
                ax.yaxis.get_offset_text().set_y(0)
                # https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))

            # f.suptitle(var_name, fontsize=16)
            plt.subplots_adjust(hspace=0.4, wspace=.3)

            # f.tight_layout()
            if file_name:
                if not os.path.exists(f'{base_dir}/{output_directory}'):
                    os.makedirs(f'{base_dir}/{output_directory}')
                file_name_ = f'{base_dir}/{output_directory}/{file_name}'
                logger.info(f'storing plot at {file_name_}')
                f.savefig(file_name_)
            plt.close('all')
            plt.rcParams.update(rcParams_bkp)
        logger.info("writing TOC")
        writer.save()

        from openpyxl import load_workbook
        wb = load_workbook(xlsx_file_name)
        ws = wb['toc']
        # ----------------------------------------------------- TV vs STB
        # ------------------ TOC ----------------
        for row, (name, desc) in enumerate(sheet_descriptions.items(), start=1):
            ws[f'A{row}'].value = desc
            ws[f'B{row}'].value = desc
            ws[f'D{row}'].value = f'=HYPERLINK("#\'{name}\'!A1", "Link")'
        ws.column_dimensions["A"].width = '23'
        ws.column_dimensions["B"].width = '63'
        ws.column_dimensions["C"].width = '10'
        # print(xlsx_file_name)
        wb.save(xlsx_file_name)


if __name__ == '__main__':
    args = setup_parser(sys.argv[1:])
    logger.info(f"Running with parameters {args}")
    if args.verbose:
        level = logging.DEBUG
        logger = logging.getLogger('ngmodel')
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    run(args)
