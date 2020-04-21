import copy
import logging
from typing import Dict, Tuple

import numpy as np
import yaml

from eam_core import Variable, ExcelVariable, Formula, FormulaModel, FormulaProcess, \
    ServiceModel, SimulationControl, defaultdict, Q_, sample_value_for_simulation
from table_data_reader import CSVHandler, OpenpyxlTableHandler

logger = logging.getLogger(__name__)


class ModuleParser(object):

    def __init__(self, version=1):
        self.version = version

    @staticmethod
    def load_definitions(doc):
        return yaml.load(doc)

    def parse_formula_process(self, definition, yaml_structure):
        result = {'params': []}

        result['id'] = definition['id']
        result['process_name'] = definition['name']
        result['model_layer'] = definition['metadata']['model_layer']

        table_file_name = yaml_structure['Metadata']['table_file_name']
        if table_file_name.endswith('.csv'):
            definitions = CSVHandler().load_definitions(filename=table_file_name)
        else:
            definitions = OpenpyxlTableHandler().load_definitions(None, filename=table_file_name)

        for value_var_tuple in definition['tableVariables']:
            var_name = value_var_tuple['value']
            for parameter_definition in definitions:
                if parameter_definition['variable'] == var_name:
                    param = {}
                    param['id'] = parameter_definition['id']
                    param['value'] = parameter_definition['ref value']
                    param['name'] = parameter_definition['variable']
                    param['unit'] = parameter_definition['unit']
                    result['params'].append(param)

        return result

    def parse_yaml_structure(self, yaml_structure) -> Dict[str, Dict]:

        yaml_process_structs = yaml_structure['Processes']
        processes = {}

        for defintion in yaml_process_structs:
            logger.info(f"parsing {defintion['name']}")

            processes[defintion['name']] = self.parse_formula_process(defintion, yaml_structure)

        return processes

    @staticmethod
    def load_module(yaml_structure) -> Dict:

        loader = ModuleParser(version=yaml_structure.get('variant', 2))

        result = {}

        result['name'] = yaml_structure['Metadata']['model_name']
        result['version'] = yaml_structure['Metadata']['model_version']
        result['processes'] = loader.parse_yaml_structure(yaml_structure)

        return result
