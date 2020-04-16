from typing import Tuple

import pandas as pd

from eam_core.util import kWh_p_J


def convert(data_frame: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str]:
    """
    Convert based on the :data:`analysis.conversion_factors`


    :param data_frame: DataFrame to convert.
    :param metric: name of the metric to process results for [use_phase_energy, use_phase_carbon]
    :return: the converted data
    """
    unit = converted_units[metric]
    return data_frame * conversion_factors[metric_conversion[metric]], unit


converted_units = {'use_phase_energy': 'MWh', 'use_phase_carbon': 'tCO2e', 'embodied_carbon': 'tCO2e'}
conversion_factors = {'J to MWh': kWh_p_J / 1000, 'gCO2e to tCO2e': 1 / 1_000_000, 'b to TB': 1e-12}
metric_conversion = {'use_phase_energy': 'J to MWh', 'use_phase_carbon': 'gCO2e to tCO2e',
                     'embodied_carbon': 'gCO2e to tCO2e', 'data_volume': 'b to TB'}