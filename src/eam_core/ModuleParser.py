import logging
import os
from typing import Dict

import yaml

from table_data_reader import CSVHandler, OpenpyxlTableHandler

logger = logging.getLogger(__name__)


class ModuleParser(object):
    param_type_map = {'p': 'proportion', 'd': 'decimal'}

    def __init__(self, yaml_file_location, version=1):
        self.version = version
        self.yaml_file_location = yaml_file_location

    @staticmethod
    def load_definitions(doc):
        return yaml.load(doc)

    def parse_formula_process(self, definition, yaml_structure):
        result = {'params': []}

        result['id'] = definition['id']
        result['name'] = definition['name']
        result['description'] = definition['metadata']['description']
        result['category'] = definition['metadata']['ui_category']

        table_file_name = yaml_structure['Metadata']['table_file_name']
        if self.yaml_file_location:
            table_file_name = self.yaml_file_location + os.path.sep + table_file_name
        if table_file_name.endswith('.csv'):
            definitions = CSVHandler().load_definitions(filename=table_file_name)
        else:
            definitions = OpenpyxlTableHandler().load_definitions(None, filename=table_file_name)
        for value_var_tuple in definition['tableVariables']:
            var_name = value_var_tuple['value']
            for parameter_definition in definitions:
                if parameter_definition['variable'] == var_name:
                    print(parameter_definition)
                    if parameter_definition['ui variable']:
                        param = {}
                        param['id'] = parameter_definition['id']
                        param['value'] = parameter_definition['ref value']
                        param['name'] = parameter_definition['variable']
                        param['unit'] = parameter_definition['unit']
                        param['description'] = parameter_definition['description']
                        param['type'] = ModuleParser.param_type_map[parameter_definition['ui variable']]
                        result['params'].append(param)
                        print(result["params"])
        print(result)
        return result

    def parse_yaml_structure(self, yaml_structure) -> Dict[str, Dict]:

        yaml_process_structs = yaml_structure['Processes']
        processes = {}

        for defintion in yaml_process_structs:
            logger.info(f"parsing {defintion['name']}")

            processes[defintion['name']] = self.parse_formula_process(defintion, yaml_structure)

        return processes

    @staticmethod
    def load_module(yaml_structure, yaml_file_location="") -> Dict:

        loader = ModuleParser(version=yaml_structure.get('variant', 2), yaml_file_location=yaml_file_location)

        result = {}

        result['name'] = yaml_structure['Metadata']['model_name']
        result['version'] = yaml_structure['Metadata']['model_version']
        result['description'] = yaml_structure['Metadata']['description']
        result['processes'] = loader.parse_yaml_structure(yaml_structure)

        return result
