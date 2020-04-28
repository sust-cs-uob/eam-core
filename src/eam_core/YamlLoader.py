import copy
import logging
from typing import Dict, Tuple

import numpy as np
import yaml
from ruamel import yaml

from eam_core import Variable, ExcelVariable, Formula, FormulaModel, FormulaProcess, \
    ServiceModel, SimulationControl, defaultdict, Q_, sample_value_for_simulation

logger = logging.getLogger(__name__)


class YamlLoader(object):
    id_map = {'process': {}, 'parameters': {}}

    def __init__(self, version=1):
        self.version = version
        self.id_map={'process': {}, 'parameters': {}}

    def create_formula_v1(self, formula_struct, yaml_structure, test=True):
        if 'ref' in formula_struct:
            ref = formula_struct['ref']

            for _fstruc in yaml_structure['Formulas']:
                if _fstruc['name'] == ref:
                    formula_struct = _fstruc
                    break

            if not 'text' in formula_struct:
                raise Exception(f'Error: Formula ref {ref} not found')

        text = formula_struct['text']

        formula = Formula(text)

        if 'test_config' in formula_struct:
            control = SimulationControl()
            control.sample_mean_value = True
            control.sample_size = 1

            d = defaultdict(lambda: 1)
            for var_name, value in formula_struct['test_config']['variables'].items():
                d[var_name] = value
            process = FormulaProcess('test', FormulaModel(formula), DSL_variable_dict=d)
            result, _ = process.evaluate(control, None)
            np.testing.assert_almost_equal(result, formula_struct['test_config']['expect'])

        return formula

    def create_formula(self, formula_struct, yaml_structure, test=True):
        if self.version == 1:
            return self.create_formula_v1(formula_struct, yaml_structure, test)
        if self.version == 2:
            return Formula(formula_struct)

    @staticmethod
    def load_definitions(doc):
        return yaml.load(doc)

    def create_variables_from_yaml_v1(self, variable_definitions, simulation_control, yaml_structure, constants) \
            -> Dict[str, Variable]:
        vars = {}

        for var_def in variable_definitions:
            if var_def['type'] == 'StaticVariable':
                vars[var_def['formula_name']] = self.create_variable(var_def['formula_name'], simulation_control,
                                                                     var_def['value'])

            if var_def['type'] == 'StaticVariables':
                for name, val in var_def['variables'].items():
                    # parse the value
                    quantity_variable = self.create_variable(name, simulation_control, val)

                    vars[name] = quantity_variable

            if var_def['type'] == 'ExcelVariable':

                matching_locations = [loc for loc in yaml_structure['Metadata']['file_locations'] if
                                      loc['file_alias'] == var_def['file_alias']]
                if len(matching_locations) == 0:
                    raise Exception(f"Could not find excel file location in with alias {var_def['file_alias']}")
                location = matching_locations[0]
                var_def['excel_file_name'] = location['file_name']
                var_def['name'] = var_def['formula_name']
                vars[var_def['formula_name']] = ExcelVariable(**var_def)
            if var_def['type'] == 'ExcelVariableSet':

                if 'file_alias' in var_def:
                    matching_locations = [loc for loc in yaml_structure['Metadata']['file_locations'] if
                                          loc['file_alias'] == var_def['file_alias']]
                    if len(matching_locations) == 0:
                        raise Exception(f"Could not find excel file location in with alias {var_def['file_alias']}")
                    location = matching_locations[0]
                else:
                    if len(yaml_structure['Metadata']['file_locations']) == 1:
                        location = yaml_structure['Metadata']['file_locations'][0]
                        logger.debug(
                            f'no file_alias defined for excel var set - implicitly using {location["file_alias"]}')
                    else:
                        raise Exception(f"Missing file alias in ExcelVariableSet")

                suffix = ""
                if 'substitution' in var_def:
                    suffix = f"_{var_def['substitution']}"

                for _var_def in var_def['variables']:

                    if type(_var_def) == str:
                        formula_name = _var_def
                        excel_var_name_stub = _var_def
                    else:
                        formula_name = list(_var_def.keys())[0]
                        excel_var_name_stub = list(_var_def.values())[0]

                    _def = {'name': formula_name, 'excel_file_name': location['file_name'],
                            'sheet_name': var_def['sheet_name'] if 'sheet_name' in var_def else None,
                            'excel_var_name': excel_var_name_stub + suffix}
                    vars[formula_name] = ExcelVariable(**_def)

            if var_def['type'] == 'Constants':
                for _var_name in var_def['variables']:
                    vars[_var_name] = constants[_var_name]
        return vars

    def create_variables_from_yaml_v2(self, variable_definitions, simulation_control, yaml_structure, constants) \
            -> Dict[str, Variable]:
        vars = {}

        for var_defs_type, var_def in variable_definitions.items():

            if var_defs_type.lower() == 'staticvariables':
                for var in var_def:
                    var_name_ = var['name']
                    value_ = var['value']
                    logger.debug(var)
                    quantity_variable = self.create_variable(var_name_, simulation_control, value_)

                    vars[var_name_] = quantity_variable

            if var_defs_type.lower() == 'tablevariables':
                for var in var_def:
                    logger.debug(f"parsing var {var['value']}")
                    var__ = {}
                    var__['excel_file_name'] = yaml_structure['Metadata']['table_file_name']
                    var__['name'] = var['value']
                    logger.debug(var)
                    vars[var__['name']] = ExcelVariable(**var__)

            if var_defs_type.lower() == 'constants':
                for conts in var_def:
                    name = conts['value']
                    if name:
                        vars[name] = constants[name]
        return vars

    def create_variables_from_yaml(self, variable_definitions, simulation_control, yaml_structure, constants) \
            -> Dict[str, Variable]:
        if self.version == 1:
            return self.create_variables_from_yaml_v1(variable_definitions, simulation_control, yaml_structure,
                                                      constants)
        if self.version == 2:
            return self.create_variables_from_yaml_v2(variable_definitions, simulation_control, yaml_structure,
                                                      constants)

    def create_variable(self, name, simulation_control, val):

        simple_quant = Q_(val)
        val = simple_quant.m
        # generate timeseries and add the unit back in
        quantity = sample_value_for_simulation(float(val), simulation_control, name, unit=simple_quant.units)
        # quantity = Q_(sample_value_for_simulation(float(val), simulation_control, name), simple_quant.units)
        logger.debug(f'creating new quantity for variable {name}')
        # assert quantity.m.index.names == ['time', 'samples']
        quantity_variable = Variable.static_variable(name, quantity)
        return quantity_variable

    def create_formula_process(self, definition, yaml_structure, sim_control, constants, prototypes):
        if 'prototype' in definition:
            pt = copy.deepcopy(prototypes[definition['prototype']])
            for k, v in definition.items():
                # // key does exist
                if k in pt:
                    if isinstance(pt[k], dict):
                        pt[k].update(v)
                    elif isinstance(pt[k], list):
                        for _v in v:
                            pt[k].append(_v)
                    else:
                        pt[k] = v
                # // key does not exist
                else:
                    pt[k] = v

            definition = pt
            del definition['type']

        input_variables = {}

        if (self.version == 1 and 'input_variables' in definition) or \
                self.version == 2:

            if self.version == 1:
                variable_definitions = definition.get('input_variables', [])
            else:
                variable_definitions = {k: v for k, v in definition.items() if
                                        k in ['tableVariables', 'staticVariables', 'constants']}

            for formula_name, var in self.create_variables_from_yaml(variable_definitions,
                                                                     sim_control,
                                                                     yaml_structure, constants).items():
                input_variables[formula_name] = var

        formula = self.create_formula(definition['formula'], yaml_structure)

        fModel = FormulaModel(formula)

        import_variables = {}
        if self.version == 1:
            for _var in definition.get('import_variables', []):

                if 'external_name' in _var:
                    incoming_name = _var['external_name']
                    import_variables[incoming_name] = {'aggregate': _var.get('aggregate', True),
                                                       'formula_name': _var.get('formula_name', _var['external_name'])}
                else:
                    assert type(_var) == str
                    incoming_name = _var
                    import_variables[incoming_name] = {'aggregate': True, 'formula_name': _var}

            export_variable_names = definition.get('export_variables')
        if self.version == 2:

            export_variable_names = [obj['value'] for obj in definition.get('exportVariables')]

            for _var in definition.get('importVariables', []):
                var_name = _var['value']
                import_variables[var_name] = {'aggregate': True, 'formula_name': var_name}

        if isinstance(export_variable_names, list):
            export_variable_names = {v: v for v in export_variable_names}

        p = FormulaProcess(definition['name'], fModel,
                           input_variables=input_variables,
                           export_variable_names=export_variable_names,
                           import_variable_names=import_variables,
                           metadata=definition.get('metadata', {}))

        return p

    def parse_yaml_structure(self, yaml_structure, sim_control) -> Tuple[Dict[str, FormulaProcess], Dict[str, str]]:
        yaml_process_structs = yaml_structure['Processes']
        yaml_constants_structs = yaml_structure.get('Constants', [])

        constants = {}

        if self.version == 1:
            # first parse constants @todo -- why parse them first?
            for formula_name, var in self.create_variables_from_yaml(yaml_constants_structs,
                                                                     sim_control,
                                                                     yaml_structure,
                                                                     constants).items():
                constants[formula_name] = var
        else:
            for const in yaml_constants_structs:
                name_ = const['name']
                const_variable = self.create_variable(name_, sim_control, const['value'])
                constants[name_] = const_variable

        processes = {}
        link_map = {}
        prototypes = {}

        for definition in yaml_process_structs:
            logger.info(f"parsing {definition['name']}")
            if 'type' in definition and definition['type'] == 'prototype':
                prototypes[definition['name']] = definition
                continue

            processes[definition['name']] = self.create_formula_process(definition, yaml_structure, sim_control,
                                                                        constants, prototypes)
            if sim_control.process_ids:
                if definition['name'] not in self.id_map['process'].keys() and 'id' in definition:
                    pid = definition['id']
                    # raises exception if the ID already exists
                    if pid in self.id_map['process'].values():
                        raise Exception("Duplicate ID for process " + definition['name'])
                    else:
                        self.id_map['process'][definition['name']] = pid
            link_map[definition['name']] = definition.get('link_to', [])
        if sim_control.process_ids:
            for definition in yaml_process_structs:
                if not ('id' in definition):
                    # If this is the first process and it has no ID, set it to 0
                    if len(self.id_map['process'].values()) == 0:
                        pid = 0
                    else:
                        pid = max(self.id_map['process'].values()) + 1  # else set it to the highest existing ID plus 1
                    self.id_map['process'][definition['name']] = pid
                    i = [ind for ind, x in enumerate(yaml_structure["Processes"]) if x["name"] == definition['name']][0]
                    # edit the YAML structure
                    yaml_structure["Processes"][i]['id'] = pid
                    # save it to the original file
                    with open(sim_control.filename, 'w') as file:
                        yaml.dump(yaml_structure, file, default_flow_style=False, Dumper=yaml.RoundTripDumper,
                                  width=4096)
                    logger.info(f"ID {pid} given to process {definition['name']}")
        return processes, link_map

    @staticmethod
    def create_service(yaml_structure, sim_control) -> ServiceModel:

        loader = YamlLoader(version=yaml_structure.get('variant', 1))

        processes, link_map = loader.parse_yaml_structure(yaml_structure, sim_control)

        s = ServiceModel(yaml_structure['Metadata']['model_name'])
        for p_name, process in processes.items():
            if link_map[p_name]:
                for linked_process in link_map[p_name]:
                    s.link_processes(process, processes[linked_process])
            else:
                s.add_process(process)

        return s

    @staticmethod
    def get_target_units(model_doc) -> Dict[str, str]:
        return model_doc['Analysis']['units']
