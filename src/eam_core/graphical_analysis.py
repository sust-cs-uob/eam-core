# coding: utf-8

# # Analysis of results

# In[1]:


# get_ipython().magic('pylab inline --no-import-all')
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd

# In[2]:
from eam_core.util import load_trace_data
from eam_core.common_graphical_analysis import plot_kind, medianprops, boxprops, whiskerprops, capprops, flierprops, \
    sum_interval

kWh_p_J = 1 / 3.60E+06

# In[3]:


# conversion_factors = {'J to MWh': kWh_p_J / 1000, 'gCO2e to tCO2e': 1 / 1_000_000, }

# In[4]:


# metric_conversion = {'use_phase_energy': 'J to MWh', 'use_phase_carbon': 'gCO2e to tCO2e',
#                      'embodied_carbon': 'gCO2e to tCO2e'}

# In[5]:


scenarios = ['baseline', 'switchover']


# metrics = ['use_phase_energy', 'use_phase_carbon', 'embodied_carbon']


# In[6]:


# # Prepare

# In[7]:


# In[8]:


# ## find all user device processes

# In[9]:
def filter_by(df, pattern_list=None, metadata=None, included_types=None):
    """
    Drop all columns from a DataFrame except if their device type and name fits.

    NOTE: If any columns are not in the simulation metadata the method returns immediately.
    @todo - why is this happening?

    :param df:
    :type df: pd.DataFrame
    :param pattern_list: if present, a list of patterns to check against. If a column fits to any of these patterns it is included
    :type pattern_list: List[str]
    :param metadata: Dict
    :type metadata:
    :param included_types: a list of (metadata) device type names to be considered for inclusion
    :type included_types: List[str]
    :return: the original DataFrame, potentially altered with new columns added and combined columns dropped
    :rtype: pd.DataFrame
    """
    # include matching only

    for col in df.columns:
        if not col in metadata:
            # we might be asked to group columns that were the result of a previous group. They would be included again
            return df
        md = metadata[col]

        # check the device type is correct
        if not included_types or 'device_type' in md and md['device_type'] in included_types:
            # check the name pattern matches
            if not pattern_list or any(ud_name.lower() in col.lower() for ud_name in pattern_list):
                continue

        df = df.drop(col, axis=1)

    return df


def exclude_by(df, pattern_list=None, metadata=None, excluded_types=None):
    # exclude matching

    for col in df.columns:
        if not col in metadata:
            # we might be asked to group columns that were the result of a previous group. They would be included again
            return df
        md = metadata[col]

        # check the device type is correct
        if not excluded_types or 'device_type' in md and md['device_type'] in excluded_types:
            # check the name pattern matches
            if not pattern_list or any(ud_name.lower() in col.lower() for ud_name in pattern_list):
                df = df.drop(col, axis=1)

    return df


def group_by(df, pattern_list=None, group_name=None, metadata=None, included_types=None):
    """
    Combine process results. The original DataFrame is returned with new columns added and combined columns dropped.
    Only columns that are included in the simulation metadata will possibly be combined into a group.

    If `included_types` is set, only columns of which the device type is included will be considered for grouping.

    :param df:
    :type df: pd.DataFrame
    :param pattern_list: if present, a list of patterns to check against. If a column fits to any of these patterns it is included
    :type pattern_list: List[str]
    :param group_name: the name of the group to create in the dataframe
    :type group_name: str
    :param metadata: Dict
    :type metadata:
    :param included_types: a list of (metadata) device type names to be considered for inclusion
    :type included_types: List[str]
    :return: the original DataFrame, potentially altered with new columns added and combined columns dropped
    :rtype: pd.DataFrame
    """
    ud_cols = []
    for col in df.columns:
        if not col in metadata:
            # a column might be the result of a previous call to group_by. We do not want to include them in future operations.
            # thus exclude them by checking that the name is present in the simulation metadata
            continue
        md = metadata[col]

        # check the device type is correct
        if not included_types or 'device_type' in md and md['device_type'] in included_types:
            # check the name pattern matches
            if not pattern_list or any(ud_name.lower() in col.lower() for ud_name in pattern_list):
                ud_cols.append(col)

    df[group_name] = df[ud_cols].sum(axis=1)
    ud_df = df.drop(ud_cols, axis=1)
    return ud_df


def group_user_devices(df, metadata=None):
    df = group_by(df, pattern_list=['TV'], group_name='TVs', metadata=metadata,
                  included_types=['Viewing Device'])
    df = group_by(df, pattern_list=['Smartphone', 'tablet', 'desktop', 'laptop', 'GamesConsole'],
                  group_name='Non-TVs', metadata=metadata, included_types=['Viewing Device'])
    return df


# In[10]:


def remove_placeholders(df):
    ph_cols = [col for col in df.columns if 'placeholder' in col]
    return df.drop(ph_cols, axis=1)


# # Plots

# In[11]:


# # Aux Functions

# In[12]:


# In[13]:


def find_user_device_processes(platform_processes, metadata):
    # find all user devices for platform
    user_devices = []
    for process in platform_processes:
        device_type = metadata[process].get('device_type', None)
        if device_type == 'Viewing Device':
            user_devices.append(process)
    return user_devices


# In[14]:


def get_platform_monthly_device_time_for_processes(platform_processes, device_time_per_month_df, metadata):
    """
    For all user devices in this platform,
    sum up all device seconds (monthly)
    convert to hours

    """
    user_devices = find_user_device_processes(platform_processes, metadata)
    device_time_per_month = sum([device_time_per_month_df[ud] for ud in user_devices])

    return device_time_per_month


# In[15]:


def plot_timeseries_of_trace_var(trace_df, var_name, process):
    trace_df.loc[var_name][process].mean(level='time').plot(ylim=(0, 2e8))


# # Plot Per Platform Results

# In[16]:


def plot_total_by_platform(df_platform_data, title='Total footprint by platform', **kwargs):
    return plot_kind(df_platform_data, title=title, **kwargs)


def get_platform_names(metadata, exclude_shared):
    """

    :param input_df:
    :param metadata:
    :param exclude_shared:
    :param device_seconds_per_month_df:
    :return:
    """
    # get platform names
    platforms = set([process_data['platform_name'] for process_name, process_data in metadata.items()])
    if exclude_shared:
        # exclude 'Shared' -> side-steps allocation problems when calculating per view
        platforms = platforms.difference(['Shared'])

    return platforms


# # Plot_combined_platform_and_total

# In[17]:


def plot_combined_platform_and_total(series_total, df_platform_data, scenario, file_name, base_dir='.', metadata=None,
                                     sharex=False, output_directory=None, **kwargs):
    """
    Horizontal boxplot chart.

    Platform boxplots at the top
    Total below.

    :param series_total:
    :param df_platform_data:
    :param scenario:
    :param file_name:
    :param base_dir:
    :param metadata:
    :param device_seconds_per_month_df:
    :param relative_to_views:
    :param sharex:
    :param kwargs:
    :return:
    """
    figsize = (10, 5)

    f, axarr = plt.subplots(2, sharex=sharex, gridspec_kw={'height_ratios': [3, 1]})

    x_limit = series_total.max()

    plot_data = plot_total_by_platform(df_platform_data, metadata=metadata, figsize=figsize, ax=axarr[0],
                                       ylabel='Platform', base_dir=base_dir, x_limit=x_limit, **kwargs)

    series_total.plot(kind='box', figsize=figsize, vert=False, ax=axarr[1],
                      label=kwargs.get('series_total_label', None), medianprops=medianprops, boxprops=boxprops,
                      whiskerprops=whiskerprops, capprops=capprops,
                      flierprops=flierprops,
                      # showmean=True
                      )  # xlim=(0, x_limit), title=f'{series_total.name}',)
    ax = axarr[1]

    formatter = tkr.FuncFormatter(lambda y, p: format(float(y), ','))
    if 'formatter' in kwargs:
        formatter = kwargs['formatter']
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel = 'All'
    ax.set_ylim(ymin=0)
    ax.get_xaxis().set_major_formatter(formatter)
    ax.grid('on', which='minor', axis='x')
    ax.grid('on', which='major', axis='x')
    plt.tight_layout()

    if file_name:
        if not os.path.exists(f'{base_dir}/{output_directory}'):
            os.makedirs(f'{base_dir}/{output_directory}')

        plt.savefig(f'{base_dir}/{output_directory}/{file_name}')
    return plot_data, series_total


def sum_over_time(df, start_date, end_date):
    if not start_date:
        start_date = df.index[0][0].date()
    if not end_date:
        end_date = df.index[-1][0].date()

    return df.loc[start_date:end_date].sum(level='samples')


def mean_of_samples(df):
    return df.mean(level='time')


def mean_per_month(df):
    return df.mean(level='samples')


def get_total_per_year(df, metadata, relative_to_views, device_seconds_per_month_df, start_date_, end_date_, unit=None,
                       aggregation='sum'):
    return get_total(df, metadata, relative_to_views, device_seconds_per_month_df, start_date_, end_date_, unit=unit,
                     aggregation=aggregation)


def get_total_monthly_average(df, metadata, relative_to_views, device_seconds_per_month_df, start_date_, end_date_,
                              unit=None, aggregation='mean'):
    return get_total(df, metadata, relative_to_views, device_seconds_per_month_df, start_date_, end_date_, unit=unit,
                     aggregation=aggregation)


def get_total(df, metadata, relative_to_views, device_seconds_per_month_df, start_date_, end_date_, unit=None,
              aggregation=None):
    """

    1. Calculate the sum of all all processes (df['total'])
    2. Slice out the bit between start and end date ( e.g. a year)
    3. Optional: aggregate. Use `aggregation='sum'` to sum to year or `aggregation='mean'` to get the monthly mean.

    :param df:
    :param metadata:
    :param relative_to_views:
    :param device_seconds_per_month_df:
    :param start_date_:
    :param end_date_:
    :return:
    """
    # prepare total
    cdf = pd.DataFrame()

    # sum over all processes (ie. STB + TV + Network, etc)
    col_name = f'Total ({unit})'
    if relative_to_views:
        col_name = f'Total per device hour ({unit}/h)'

    cdf[col_name] = df.sum(axis=1)

    if relative_to_views:
        device_hours = get_platform_monthly_device_time_for_processes(metadata.keys(), device_seconds_per_month_df,
                                                                      metadata)
        cdf[col_name] = cdf[col_name] / device_hours

    # slice out year
    timeframe_limited_total_series = cdf[col_name].loc[start_date_:end_date_]

    series_total = timeframe_limited_total_series

    # sum over year
    if aggregation is not None and aggregation == 'sum':
        series_total = sum_interval(timeframe_limited_total_series)
        series_total.rename(f'{col_name} -- per year', inplace=True)
    if aggregation is not None and aggregation == 'mean':
        series_total = mean_per_month(timeframe_limited_total_series)

        series_total.rename(f'{col_name} -- monthly', inplace=True)

    return series_total, col_name


# In[18]:
def get_platform_total(df, device_hours_df=None, metadata=None, relative_to_views=False):
    """
    0. find all platforms in the dataset
    1. define new dataframe with
    2. for each platform:
        - find all processes in this platform
        - sum up the footprint over all processes in the platform

        if relative to views
        - get the total sum of device hours on user devices in this platform
        - divide the total platform footprint by the user hours in this platform

    The shared impact is ignored.


    """
    # get platform names
    platforms = set([process_data.get('platform_name', 'no platform') for process_name, process_data in metadata.items()])

    if relative_to_views:
        # @todo allocate across platforms
        platforms = platforms.difference(['Shared'])

    cdf = pd.DataFrame()
    # aggregate processes by platform
    for platform in platforms:
        # find all processes from this platform
        platform_processes = [process_name for process_name, process_data in metadata.items() if
                              process_data.get('platform_name',"") == platform]

        cdf[platform] = df[platform_processes].sum(axis=1)

        if relative_to_views:
            device_hours = get_platform_monthly_device_time_for_processes(platform_processes, device_hours_df,
                                                                          metadata)
            cdf[platform] = cdf[platform] / device_hours

    return cdf


# ['terr', 'iplayer', 'sat', 'internal']
def get_platform_data_and_plot(input_data, platform, metadata=None, **kwargs):
    return plot_kind(get_platform_devicetype_aggregate(input_data, metadata, platform), **kwargs)


def get_platform_devicetype_aggregate(input_data, metadata, platform):
    """
    For a given platform
    For all device type categories
    sum up footprint over all processes in the category

    :param input_data:
    :param metadata:
    :param platform:
    :return:
    """
    platform_processes = [process_name for process_name, process_data in metadata.items() if
                          process_data.get('platform_name', "") == platform]
    df = pd.DataFrame()
    # re-categorise all processes by device_types ("user device", "access network", etc)
    device_types = defaultdict(list)
    for process in platform_processes:
        device_type = metadata[process].get('device_type', process)
        device_types[device_type].append(process)

    for device_type, processes in device_types.items():
        df[device_type] = input_data[processes].sum(axis=1)

    data = df[list(device_types.keys())]
    return data


# In[ ]:
def plot_platform_process_annual_total(common_args, end_date, load_data, metadata, start_date, writer,
                                       platform='iPlayer', scenario='baseline'):
    """
    Plot and export to excel

    annual totals of the platform processes aggregated by device types

    """
    # aggregate data by device types
    data, unit = load_data()
    iplayer_devicetype_data = get_platform_devicetype_aggregate(data, metadata, platform)
    # sum up to annual
    annual_iplayer_devicetype_data = sum_interval(iplayer_devicetype_data, start_date, end_date)
    # dump to excel
    annual_iplayer_devicetype_data.quantile([.25, .5, .75]).to_excel(writer, f'{platform} per a')
    # plot
    plot_kind(annual_iplayer_devicetype_data, file_name=f'{scenario}_{platform}_annual_total_{unit}.pdf',
              figsize=(10, 5),
              title=f'{platform} Annual Total ({unit})', **common_args)


def plot_platform_process_per_device_hour(common_args, end_date, load_data, metadata, start_date, writer,
                                          platform='iPlayer', scenario='baseline', base_dir=None, output_directory=None):
    """
    Plot and export to excel

    annual totals of the platform processes aggregated by device types per device hour

    1. get the processes in a platform
    2. get the viewer hours for the platform
    3. divide one by the other

    """
    # aggregate data by device types
    data, unit = load_data()
    platform_processes = [process_name for process_name, process_data in metadata.items() if
                          process_data.get('platform_name', "") == platform]
    df = pd.DataFrame()
    # re-categorise all processes by device_types ("user device", "access network", etc)

    for process in platform_processes:
        df[process] = data[process]

    device_secs_per_month_df = load_trace_data(output_directory, 'duration', base_dir=base_dir)
    device_hours_per_month_df = device_secs_per_month_df / 3600

    for process in df.columns:
        df[process] = df[process] / device_hours_per_month_df[process]

    # dump to excel
    df.quantile([.25, .5, .75]).to_excel(writer, f'{platform} per device hour')
    # plot
    plot_kind(df, file_name=f'{scenario}_{platform}_per_device_hour_{unit}.pdf',
              figsize=(10, 5),
              title=f'{platform} per device hour ({unit}/viewer-hour)', **common_args)


