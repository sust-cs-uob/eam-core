import unittest
from ruamel import yaml
from tests.directory_test_controller import set_cwd_to_script_dir, return_to_base_cwd

from eam_core.ModuleParser import ModuleParser


class ModuleParserTestCase(unittest.TestCase):
    doc = u"""
syntax_variant: 2

Processes:
- name: CDN
  order: 1
  metadata:
    ui_category: Datacentre
    description: |
        A CDN is a
        system that accelerates access
  tableVariables:
  - value: energy_intensity_network
  - value: carbon_intensity
  formula: |
    energy = energy_intensity_network * data_volume;
    carbon = energy * carbon_intensity;
    return energy

  id: 0

Metadata:
  status: testing
  process_map: foo
  model_name: CI_model
  description: |
    bla bla
  model_version: 0.0.1
  table_file_name: data/ci_v2_data.xlsx
  datasource_version_hint: 2
  comparison_variable: energy
  start_date: 2019-01-01
  end_date: 2036-01-01
  sample_size: 1
  sample_mean: true
  individual_process_graphs_variable: energy
  analysis_configs:
  - name: ci
    named_plots:
#        - all_area
    - individual_processes
    - input_vars
    individual_process_graphs:
    - Laptop
    standard_plots:
    - process_tree
Analysis:
  result_variables:
  - energy
  scenarios:
  - S1
  - default

  numerical:
  - energy
  units:
  - endswith: energy
    to_unit: GWh
  - __eq__: carbon
    to_unit: Mt
  - __eq__: data_volume
    to_unit: TB
  - __eq__: time
    to_unit: kyear
  - __eq__: result
    to_unit: GWh

  plots:
  - name: individual_processes
    variable: energy
    kind: grid
    title: Monthly Energy Consumption per Process
    xlabel: Time
    ylabel: Energy per Month [{unit}]
Constants:
- name: ref_duration
  value: 2628000s
"""

    def test_variable_data(self):
        """
        test variables are read correctly
        - id
        - name
        - unit
        - description
        - only include those mentioned in table variables
        - only include table variables that are marked with 'ui' in the table (excludes energy_intensity)
        :return:
        """
        cwd = set_cwd_to_script_dir()
        module = ModuleParser.load_module(yaml.safe_load(ModuleParserTestCase.doc, ))
        process = next(iter(module['processes'].values()))
        params = process['params']
        # print(params)
        param_map = {v['name']: v for v in params}
        # print(param_map)
        assert set(param_map.keys()) == {'carbon intensity'}
        assert param_map['carbon intensity']['value'] == 0.5
        assert param_map['carbon intensity']['id'] == 4
        assert param_map['carbon intensity']['unit'] == 'kg/kWh'
        assert param_map['carbon intensity']['type'] == 'proportion'
        return_to_base_cwd(cwd)


    def test_process_data(self):
        """
        test that process info is read correctly
        - id
        - name
        - description
        :return:
        """
        cwd = set_cwd_to_script_dir()
        module = ModuleParser.load_module(yaml.load(ModuleParserTestCase.doc))

        process = next(iter(module['processes'].values()))

        assert process['id'] == 0
        assert process['name'] == 'CDN'
        assert process['category'] == 'Datacentre'
        return_to_base_cwd(cwd)

    def test_model_metadata(self):
        """
        test that model metadata is read correctly
        - description
        - version
        - name
        :return:
        """

        cwd = set_cwd_to_script_dir()
        module = ModuleParser.load_module(yaml.safe_load(ModuleParserTestCase.doc))
        assert module['name'] == 'CI_model'
        assert module['version'] == '0.0.1'
        return_to_base_cwd(cwd)


if __name__ == '__main__':
    unittest.main()
