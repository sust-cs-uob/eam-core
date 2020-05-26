import logging
import math
import os
from functools import partial

import matplotlib
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import simplejson as json
from cycler import cycler
from matplotlib import ticker as tkr, pyplot as plt

logger = logging.getLogger(__name__)

default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))


def plot_grid(df, file_name, xlabel, ylabel, title, base_dir, output_scenario_directory, **kwargs):
    data = df.ix[:, df.groupby(level=['time']).quantile(.75).max().sort_values(ascending=False).index]

    maxima = data[data.columns[0]].groupby(level=['time']).quantile(.75).mean()
    minima = data[data.columns[-1]].groupby(level=['time']).quantile(.75).mean()

    norm = matplotlib.colors.LogNorm(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis_r)

    plt.close('all')

    # sort
    # data = data.ix[:, data.groupby(level=['time']).quantile(.75).max().sort_values(ascending=False).index]

    n = len(data.columns)
    # fig = plt.figure(figsize=(12,n*0.75))
    rows = math.ceil(n / 3)
    f, axarr = plt.subplots(rows, 3, sharex='col', sharey='row')
    f.set_size_inches(10, n * 1)

    # joint max of 75th percentile
    # yhigh = data.abs().groupby(level=['time']).quantile(.75).values.max() * 1.1

    # joint min of 25th percentile
    # ylow = data.abs().groupby(level=['time']).quantile(.75).values.max() * 1.1

    for i, ax in enumerate(axarr.flat):
        if i == len(data.columns):
            break
        ax.set_title(data.columns[i])
        process_name = data.columns[i]
        #     print(f'{i}: {process_name}')

        process_data = data[process_name]
        mean_ = process_data.mean(level='time')

        grouped_ = process_data.groupby(level=['time'])

        low = grouped_.quantile(.25)
        high = grouped_.quantile(.75)

        mean = high.mean()
        logger.debug(mean)
        mean_.plot(ax=ax,
                   kind='line',
                   legend=False,
                   linewidth=1,
                   color=mapper.to_rgba(mean)
                   )

        low.plot(ax=ax, color='k', alpha=0.2, linestyle=':')
        high.plot(ax=ax, color='k', alpha=0.2, linestyle=':')
        ax.fill_between(mean_.index.get_level_values(0),
                        low,
                        high,
                        facecolor='k',
                        #                      hatch="+",
                        edgecolor="k", linewidth=0.1,
                        alpha=0.1, interpolate=True, linestyle='-')
        ylim = high.max()

        if i % 3 == 0:
            ax.set_ylim([0, ylim * 1.3])
        #     print(f'{ylim}')

        ax.set_xlabel('')

    f.suptitle(title)
    f.text(0.5, -0.0, xlabel, ha='center')
    f.text(-0.0, 0.5, ylabel, va='center', rotation='vertical')
    # data.abs().sum(axis=1).groupby(level=['time']).quantile(.25).plot(ax=ax, color='red')
    f.subplots_adjust(bottom=0.2)
    f.subplots_adjust(top=.8)
    f.subplots_adjust(right=.9)
    f.subplots_adjust(left=0.2)
    # plt.tight_layout()
    if file_name:
        if not os.path.exists(f'{base_dir}/{output_scenario_directory}'):
            os.makedirs(f'{base_dir}/{output_scenario_directory}')
        file_name_ = f'{base_dir}/{output_scenario_directory}/{file_name}'
        logger.info(f'storing plot at {file_name_}')
        f.savefig(file_name_)
    plt.close('all')


def plot_process_with_input_vars(model, sim_control, data, file_name, output_scenario_directory, unit, base_dir='.',
                                 analysis_config=None, image_filetype=None):
    all_vars = model.collect_input_variables()

    plot_input_var_for_process_f = partial(plot_input_var_for_process, base_dir=base_dir, data=data,
                                           file_name=file_name,
                                           output_scenario_directory=output_scenario_directory, sim_control=sim_control,
                                           analysis_config=analysis_config, unit=unit, image_filetype=image_filetype)

    if False:
        num_cores = multiprocessing.cpu_count()
        p = multiprocessing.Pool(num_cores)
        results = p.map(plot_input_var_for_process_f, sorted(all_vars.items()))
    else:
        results = []
        for proc_name_vars_tupel in sorted(all_vars.items()):
            results.append(plot_input_var_for_process_f(proc_name_vars_tupel))

    iv_dfs = dict((x, y) for x, y in results if x)
    return iv_dfs


def plot_input_var_for_process(proc_name_vars_tupel, base_dir=None, data=None, file_name=None,
                               output_scenario_directory=None, sim_control=None, analysis_config=None, unit=None,
                               image_filetype=None):
    """

    :param proc_name_vars_tupel: (process name, process variables)
    :type proc_name_vars_tupel:
    :param base_dir:
    :type base_dir:
    :param data:
    :type data:
    :param file_name:
    :type file_name:
    :param output_scenario_directory:
    :type output_scenario_directory:
    :param sim_control:
    :type sim_control:
    :param analysis_config:
    :type analysis_config:
    :return:
    :rtype:
    """
    proc_name, vars = proc_name_vars_tupel[0], proc_name_vars_tupel[1]

    # print (analysis_config)
    if analysis_config is not None \
        and 'individual_process_graphs' in analysis_config \
        and not proc_name in analysis_config['individual_process_graphs']:
        logger.debug(f"skipping individual process plot for {proc_name}")
        return (False, None)

    # convert the pint quantity data to a data frame
    p_df = None
    units = {}
    for var_name, var in sorted(vars.items()):

        if 'energy' == var_name:
            continue
            #         print(proc_name, var_name, var)
        #  need to convert
        if p_df is None:
            p_df = pd.DataFrame(data=var.pint.m)
            p_df.columns = [var_name]
        else:
            p_df[var_name] = var.pint.m

        units[var_name] = var.pint.units

    if not isinstance(p_df.index, pd.MultiIndex):
        p_df.index = sim_control._df_multi_index

    # a cope of the data to return from the function
    p_df_copy = p_df.copy()
    # print(units)
    # normalise all variables to [0,1]
    for column in p_df.columns:
        # constants are normalised to 0.5
        if p_df[column].std() == 0:
            p_df[column] = (p_df[column] - p_df[column].mean()) + 1 / 2
        else:
            p_df[column] = p_df[[column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    process_name = proc_name

    plt.close('all')

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if process_name in data.columns:
        process_data = data[process_name]
        mean_ = process_data.mean(level='time')
        grouped_ = process_data.groupby(level=['time'])
        low = grouped_.quantile(.25)
        high = grouped_.quantile(.75)
        mean_.plot(ax=ax, kind='line',
                   legend=False,
                   linewidth=1,
                   color='k', alpha=.3,
                   marker="x"
                   )
        ax.set_ylabel(unit)

        low.plot(ax=ax, color='k', alpha=0.2, linestyle=':')
        high.plot(ax=ax, color='k', alpha=0.2, linestyle=':')
        ax.fill_between(mean_.index.get_level_values(0),
                        low,
                        high,
                        facecolor='k',
                        #                      hatch="+",
                        edgecolor="k", linewidth=0.1,
                        alpha=0.1, interpolate=True, linestyle='-')

    # ax.set_ylim(bottom=0)

    color = cm.brg(np.linspace(0, 1, len(p_df.columns)))

    from random import choices
    linestyle = choices(['-', '--', ':'], k=len(p_df.columns))
    p_df.mean(level='time').plot.line(ax=ax, secondary_y=True, linewidth=1, color=color, style=linestyle)
    # linestyle = linestyle
    ax.right_ax.set_ylim(0, 1)

    handles, labels = ax.right_ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    ncol=math.ceil(len(p_df.columns) / len(p_df.columns) / 3), shadow=True)
    # text = ax.text(-0.3, 1, "test", transform=ax.transAxes)
    fig.suptitle(file_name)
    if file_name:
        if not os.path.exists(f'{base_dir}/{output_scenario_directory}/input_var_plots'):
            os.makedirs(f'{base_dir}/{output_scenario_directory}/input_var_plots')
        file_name_ = f'{base_dir}/{output_scenario_directory}/input_var_plots/{file_name}_{process_name}.{image_filetype}'
        logger.info(f'storing plot at {file_name_}')
        fig.savefig(file_name_, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')

    return (proc_name, p_df_copy)


def plot_kind(df, base_dir='.', output_scenario_directory=None, file_name=None, title=None, x_limit=None, ax=None,
              xlabel=None, figsize=(10, 7), ylabel=None, kind='box', **kwargs) -> pd.DataFrame:
    if kind == 'box':
        return plot_box(ax, df, figsize, file_name, kind, x_limit, xlabel, ylabel, base_dir=base_dir, title=title,
                        output_scenario_directory=output_scenario_directory, **kwargs)
    elif kind == 'bar':
        return plot_bar(df, file_name, xlabel, ylabel, title=title, base_dir=base_dir,
                        output_scenario_directory=output_scenario_directory, **kwargs)
    elif kind == 'area':
        return plot_area(df, file_name, xlabel, ylabel, title=title, base_dir=base_dir,
                         output_scenario_directory=output_scenario_directory, **kwargs)
    elif kind == 'grid':
        return plot_grid(df, file_name, xlabel, ylabel, title=title, base_dir=base_dir,
                         output_scenario_directory=output_scenario_directory, **kwargs)


def plot_bar(df, file_name, xlabel, ylabel, figsize=(10, 30), title=None, base_dir='.', output_scenario_directory=None,
             **kwargs):
    plt.close('all')

    import copy
    rcParams_bkp = copy.deepcopy(plt.rcParams)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.linewidth'] = .2
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['lines.linewidth'] = .1
    plt.rcParams['hatch.linewidth'] = 0.3

    # fig = plt.figure()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    title = f'{title} ({kwargs["start_date"]} - {kwargs["end_date"]})'
    import matplotlib.font_manager as fm

    prop = fm.FontProperties()
    ax.set_title(title, fontproperties=prop, size='large')

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(prop)
        label.set_fontsize('medium')  # Size here overrides font_prop

    ax.set_ylabel('y', fontproperties=prop)
    ax.set_xlabel('x', fontproperties=prop)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    df = sum_interval(df, kwargs['start_date'], kwargs['end_date'])
    df.reindex(df.mean().sort_values().index, axis=1)
    df.T.plot(ax=ax, kind='barh', legend=False, color='#0f0f0f80', edgecolor='k', align='center', width=0.5,
              linewidth=0.5)

    if kwargs.get('use_hatch', False):
        bars = ax.patches
        patterns = ['///', '--', '...', '\///', 'xxx', '\\\\']
        hatches = [p for p in patterns for i in range(len(df))]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    from textwrap import wrap
    ax.set_yticklabels(['\n'.join(wrap(l.get_text(), 10)) for l in ax.get_yticklabels()])
    plt.tight_layout()
    if file_name:
        if not os.path.exists(f'{base_dir}/{output_scenario_directory}'):
            os.makedirs(f'{base_dir}/{output_scenario_directory}')
        file_name_ = f'{base_dir}/{output_scenario_directory}/{file_name}'
        logger.info(f'storing plot at {file_name_}')
        fig.savefig(file_name_)
    # restore
    plt.rcParams.update(rcParams_bkp)
    return df


def plot_area(df, file_name, xlabel, ylabel, figsize=(10, 7), title=None, base_dir='.', output_scenario_directory=None,
              **kwargs):
    plt.close('all')

    import copy
    rcParams_bkp = copy.deepcopy(plt.rcParams)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.linewidth'] = .2
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['lines.linewidth'] = .1
    plt.rcParams['hatch.linewidth'] = 0.3
    mean_data = kwargs['mean_data'] if 'mean_data' in kwargs else df
    # color = cm.viridis(np.linspace(0, 1, len(df.columns)))
    # plt.rcParams['axes.prop_cycle'] = cycler('color', color) * cycler('linestyle', ['-', '--', ':', '-.'])

    # plt.rcParams.update(rcParams_bkp)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    title = f'{title} ({kwargs["start_date"]} - {kwargs["end_date"]})'

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    colourmap = 'tab20'
    if len(df.columns) > 20:
        colourmap = 'prism'

    # df.reindex_axis(df.mean().sort_values().index, axis=1)
    mean_data.mean(level='time').plot(ax=ax,
                                      kind='area',
                                      # legend=False,
                                      linewidth=0.5,
                                      title=title,
                                      colormap=colourmap)

    low = df.sum(axis=1).groupby(level=['time']).quantile(.25)
    low.plot(ax=ax, color='k', alpha=0.2, linestyle='-.', label="25th CI")

    high = df.sum(axis=1).groupby(level=['time']).quantile(.75)
    high.plot(ax=ax, color='k', alpha=0.2, linestyle='--', label="75th CI")

    ylim = high.max()
    ax.set_ylim([0, ylim * 1.1])
    plt.fill_between(df.mean(level='time').index.get_level_values(0),
                     low,
                     high,
                     facecolor='k',
                     hatch="+",
                     edgecolor="k", linewidth=0.1,
                     alpha=0.1, interpolate=True, )

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    ncol=math.ceil(len(df.columns) / len(df.columns) / 3), shadow=True)

    # ax.legend(loc='center left', bbox_to_anchor=(.8, 0.5),
    #           ncol=1, fancybox=True, shadow=True)

    # bars = ax.patches
    # patterns = ['///', '--', '...', '\///', 'xxx', '\\\\']
    # hatches = [p for p in patterns for i in range(len(df))]
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)

    # from textwrap import wrap
    # ax.set_yticklabels(['\n'.join(wrap(l.get_text(), 10)) for l in ax.get_yticklabels()])

    if file_name:
        if not os.path.exists(f'{base_dir}/{output_scenario_directory}'):
            os.makedirs(f'{base_dir}/{output_scenario_directory}')
        file_name_ = f'{base_dir}/{output_scenario_directory}/{file_name}'
        logger.info(f'storing plot at {file_name_}')
        fig.savefig(file_name_, bbox_extra_artists=(lgd,), bbox_inches='tight')

    # restore
    plt.rcParams.update(rcParams_bkp)
    return df


def plot_box(ax, df, figsize, file_name, kind, x_limit, xlabel, ylabel, title=None, base_dir=None,
             output_scenario_directory=None, **kwargs):
    """
    Returns the df arg, just sorted by the mean value

    :param df:
    :param base_dir:
    :param output_scenario_directory:
    :param file_name:
    :param title:
    :param x_limit:
    :param ax:
    :param xlabel:
    :param figsize:
    :param ylabel:
    :param formatter:
    :param kwargs:
    :return:
    """
    if file_name:
        plt.close('all')
    # order columns by their mean value
    # http://stackoverflow.com/questions/17712163/pandas-sorting-columns-by-their-mean-value
    sorted_data = df.reindex(df.mean().sort_values().index, axis=1)
    if not x_limit:
        x_limit = 0
    x_limit = max(get_max(sorted_data), x_limit) * 1.1
    title = f'{title} ({kwargs["start_date"]}-{kwargs["end_date"]})'
    plot_kwargs = {}
    if kind == 'box':
        plot_kwargs = {'vert': False, 'medianprops': medianprops,
                       'boxprops': boxprops,
                       'whiskerprops': whiskerprops,
                       'capprops': capprops,
                       'flierprops': flierprops}
    ax = sorted_data.plot(kind=kind, figsize=figsize, xlim=(0, x_limit), title=title, ax=ax, **plot_kwargs)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ymin=0)
    if x_limit > 1:
        formatter = tkr.FuncFormatter(lambda y, p: format(int(y), ','))
        ax.get_xaxis().set_major_formatter(formatter)
    ax.grid('on', which='minor', axis='x')
    ax.grid('on', which='major', axis='x')
    plt.tight_layout()
    if file_name:
        if not os.path.exists(f'{base_dir}/{output_scenario_directory}'):
            os.makedirs(f'{base_dir}/{output_scenario_directory}')

        plt.savefig(f'{base_dir}/{output_scenario_directory}/{file_name}')
    return sorted_data


def load_metadata(output_directory, base_dir='.'):
    metadata = None
    with open(f'{base_dir}/{output_directory}/process_metadata.json') as f:
        metadata = json.load(f)
    return metadata


medianprops = {'color': 'magenta', 'linewidth': 1}
boxprops = {'color': 'black', 'linestyle': '-', 'linewidth': .5}
whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': .5}
capprops = {'color': 'black', 'linestyle': '-', 'linewidth': .5}
flierprops = {'color': 'black', 'marker': '.', 'alpha': 0.3}


def get_max(dfa, mean=False):
    if mean:
        dfa = dfa.mean()
    return dfa.max().max()


def sum_interval(df, start_date=None, end_date=None):
    """
    Sums up all values of the same sample index across all months in the data. For example, all values across the year to
    give the annual total.

    Result is an array of values of sample size length.
    :param df:
    :param start_date:
    :param end_date:
    :return:
    """
    if not start_date:
        start_date = df.index[0][0].date()
    if not end_date:
        end_date = df.index[-1][0].date()
    df = df.loc[start_date:end_date].sum(level='samples')
    return df
