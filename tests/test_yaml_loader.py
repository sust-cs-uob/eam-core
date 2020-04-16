import datetime
import unittest
from typing import Dict

import numpy as np
import yaml

from eam_core import SimulationControl, Variable
from eam_core.YamlLoader import YamlLoader


class YamlLoader_v1(unittest.TestCase):
    doc = u"""    
Processes:    
  - name: test formula process
    formula:
      text: |
        energy = energy_intensity * data_volume * custom_coefficient
        return energy
      test_result: 1
    input_variables:
      - formula_name: energy_intensity
        type: StaticVariable
        value: 5
      - formula_name: custom_coefficient
        type: ExcelVariable    
        file_alias: test_location
        sheet_name: Sheet1
        excel_var_name: a
    import_variables:
      - energy_intensity
      - data_volume        
    export_variables:
      energy_intensity: test_energy_intensity    
      data_volume: test_data_volume    
    link_to: 
      - process a
  - name: process stub
Formulas:
  - name: unused intensity-volume formula  
    text: |
      energy = energy_intensity * data_volume
      return energy
    test_result: 1
Metadata:
  model_name: test model
  file_locations:
    - file_alias: test_location   
      file_name: data/test_data.xlsx
      type: google_spreadsheet
      google_id_alias: scenarios_public_params
    - file_alias: dummy_location
      file_name: _tmp_data/dummy.xlsx
  start_date: 2016-01-01 
  end_date: 2032-01-01
  sample_size: 2
  sample_mean: False 
"""

    def test_yaml_parsing(self):
        yaml_def = yaml.load(YamlLoader_v1.doc)
        assert len(yaml_def) == 3
        # print(yaml_def)
        process_def = yaml_def['Processes'][0]
        # print(process_def)
        assert process_def['formula']['test_result'] == 1

        assert process_def['name'] == 'test formula process'

        assert len(process_def['import_variables']) == 2

        assert process_def['input_variables'][0]['formula_name'] == 'energy_intensity'
        assert process_def['input_variables'][0]['type'] == 'StaticVariable'
        assert process_def['input_variables'][0]['value'] == 5
        assert process_def['input_variables'][1]['formula_name'] == 'custom_coefficient'
        assert process_def['input_variables'][1]['type'] == 'ExcelVariable'
        assert process_def['input_variables'][1]['sheet_name'] == 'Sheet1'
        assert process_def['input_variables'][1]['excel_var_name'] == 'a'
        assert process_def['input_variables'][1]['file_alias'] == 'test_location'

        assert process_def['import_variables'][1] == 'data_volume'

        assert process_def['export_variables']['energy_intensity'] == 'test_energy_intensity'
        assert process_def['export_variables']['data_volume'] == 'test_data_volume'
        assert process_def[
                   'formula']['text'] == 'energy = energy_intensity * data_volume * custom_coefficient\nreturn energy\n'
        # assert process_def[                   'formula']['name'] == 'test formula'
        assert process_def['link_to'][0] == 'process a'

        metadata_struct = yaml_def['Metadata']
        # print(metadata_struct)
        assert len(metadata_struct['file_locations']) == 2
        assert metadata_struct['model_name'] == 'test model'
        # print(metadata_struct['start_date'])
        assert metadata_struct['start_date'] == datetime.date(2016, 1, 1)

    def test_excelvar(self):
        yaml_def = yaml.load(YamlLoader_v1.doc)
        metadata_struct = yaml_def['Metadata']

        formula_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 2
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(formula_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])
        assert isinstance(vars['custom_coefficient'], Variable)

    def test_excelvar_set(self):
        doc = u"""
        Processes:        
          - name: a
            formula:
              text: |
               return a + b
            input_variables:
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1
                variables:
                  - a
                  - b                                          
        Metadata:
          model_name: test model
          file_locations:
            - file_alias: test_location
              file_name: tests/data/test_data.xlsx            
        """

        yaml_def = yaml.load(doc)

        formula_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 2
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(formula_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])
        assert isinstance(vars['a'], Variable)

    def test_constants(self):
        doc = u"""
        Processes:        
          - name: pi square
            formula:
              text: |
               return pi * pi
            input_variables:                          
              - type: Constants
                variables:
                  - pi
        Constants:
          - type: StaticVariables
            variables:
              pi: 3.14                              
        Metadata:
          model_name: test model                      
        """

        sim_control = SimulationControl()

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        # for node in s.process_graph.nodes_iter():
        #     if node.name in ['a', 'b']:
        #         assert s.process_graph.out_edges(node) == []
        #
        #     if node.name == 'c':
        #         linked_nodes = set()
        #         for (_, p) in s.process_graph.out_edges(node):
        #             linked_nodes.add(p.name)
        #         assert linked_nodes == {'a', 'b'}

    def test_complex_variables(self):
        doc = u"""
        Processes:        
          - name: DTT TV Amplifier
            import_variables:
              - formula_name: service_on_time_per_day_mins
                external_name: dtt_tv_amplifier_bbc_servcie_time_mins_per_day
              - formula_name: number
                external_name: num_hh_dtt_tv_amplifer
            link_to:
               - process a
    
            formula:
              ref: TV viewing device
            
            input_variables:
              - type: StaticVariables
                variables:
                  passive_standby_time_per_day_mins: 0
                  service_active_standby_time_per_day_mins: 0
                  total_active_standby_time_per_day_mins: 0
                  total_on_time_per_day_mins: 1440
          
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1
                substitution: dtt
                variables:
                  - power
                  - bitrate
                  - power_active_standby
                  - power_passive_standby               
          
          - name: a
            formula:
              text: |
               return a + b
            input_variables:
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1
                variables:
                  - a
                  - b                                          
        Metadata:
          model_name: test model
          file_locations:
            - file_alias: test_location
              file_name: data/test_data.xlsx            
        """

        yaml_def = yaml.load(doc)

        formula_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 2
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(formula_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])

        assert isinstance(vars['passive_standby_time_per_day_mins'], Variable)

    def test_staticvar_set(self):
        doc = u"""
        Processes:        
          - name: a
            formula:
              text: |
               return a + b
            input_variables:
              - type: StaticVariables
                variables:          
                  a: 1
                  b: 2                                       
        """

        yaml_def = yaml.load(doc)

        process_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(process_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])
        assert vars['a'].data_source.get_value('test', sim_control) == 1
        assert vars['b'].data_source.get_value('test', sim_control) == 2

    def test_excelvar_set_with_sub(self):
        doc = u"""
        Processes:        
          - name: a
            formula:
              text: |
               return a + b
            input_variables:
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1
                substitution: test
                variables:
                  - a
                  - b                                          
        Metadata:
          model_name: test model
          file_locations:
            - file_alias: test_location
              file_name: tests/data/test_data.xlsx            
        """

        yaml_def = yaml.load(doc)

        formula_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(formula_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])
        assert isinstance(vars['a'], Variable)
        v = vars['a'].data_source.get_value('test', sim_control)
        # print(v)
        assert v == 1.5

    def test_excelvar_set_with_sub_and_excel_var_stub(self):
        doc = u"""
        Processes:        
          - name: a
            formula:
              text: |
               return a + b
            input_variables:
              - type: ExcelVariableSet
                file_alias: test_location
                sheet_name: Sheet1
                substitution: test
                variables:
                  a: a_stub
                  b: b_stub                                          
        Metadata:
          model_name: test model
          file_locations:
            - file_alias: test_location
              file_name: tests/data/test_data.xlsx            
        """

        yaml_def = yaml.load(doc)

        formula_def = yaml_def['Processes'][0]

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        vars: Dict[str, Variable] = YamlLoader().create_variables_from_yaml(formula_def.get('input_variables', []),
                                                                            sim_control, yaml_def, [])
        assert isinstance(vars['a'], Variable)
        v = vars['a'].data_source.get_value('test', sim_control)
        # print(v)
        assert v == 1.5

    def test_linkto_declaration(self):
        doc = u"""
    Processes:        
      - name: a
        formula:
          text: |
           return 2            
      - name: b
        formula:
          text: |
           return 2            
      - name: c
        formula:
          text: |
            return 3                        
        link_to: 
          - a
          - b
    Metadata:
      model_name: test
    """

        sim_control = SimulationControl()

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        for node in s.process_graph.nodes_iter():
            if node.name in ['a', 'b']:
                assert s.process_graph.out_edges(node) == []

            if node.name == 'c':
                linked_nodes = set()
                for (_, p) in s.process_graph.out_edges(node):
                    linked_nodes.add(p.name)
                assert linked_nodes == {'a', 'b'}

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_import_declaration(self):
        doc = u"""
Processes:        
  - name: process a
    formula:
      text: |
       return a + b + c
    input_variables:
      - formula_name: a
        type: StaticVariable
        value: 6
    import_variables:
      - b        
      - c        
  - name: process b
    formula:
      text: |
        c = 3
        b = 2                
    export_variables:      
      - b   
      - c 
    link_to: 
      - process a
Metadata:
  model_name: test
"""

        sim_control = SimulationControl()

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['process a'], [11.], rtol=.1)

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_import_alias_declaration(self):
        doc = u"""
Processes:        
  - name: process a
    formula:
      text: |
       return a + c
    input_variables:
      - formula_name: a
        type: StaticVariable
        value: 6
    import_variables:
      - formula_name: c        
        external_name: b
  - name: process b
    formula:
      text: |
        b = 2                
    export_variables:      
      b: b    
    link_to: 
      - process a
Metadata:
  model_name: test
"""

        sim_control = SimulationControl()

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['process a'], [8.], rtol=.1)

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_export_alias_declaration(self):
        doc = u"""
Processes:        
  - name: process a
    formula:
      text: |
       return a + b
    input_variables:
      - formula_name: a
        type: StaticVariable
        value: 6
    import_variables:
      - b        
  - name: process b
    formula:
      text: |
        c = 2
        a = 2                
    export_variables:      
      c: b
      a: d 
    link_to: 
      - process a
Metadata:
  model_name: test
"""

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        sim_control.use_time_series = False

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['process a'], [8.], rtol=.1)

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_multiple_formula_process(self):
        doc = u"""
Processes:        
  - name: process a
    formula:
      ref: basic intensity-volume formula              
    input_variables:
      - formula_name: energy_intensity
        type: StaticVariable
        value: 6
    import_variables:    
      - data_volume
  - name: process b
    formula:
      ref: basic intensity-volume formula      
    input_variables:
      - formula_name: energy_intensity
        type: StaticVariable
        value: 5
      - formula_name: data_volume
        type: StaticVariable
        value: 4
    export_variables:      
      data_volume: data_volume
    test_result: 1
    link_to: 
      - process a
Formulas:
  - name: basic intensity-volume formula  
    text: |
      energy = energy_intensity * data_volume
      return energy
    test_result: 1
Metadata:
  model_name: test
"""

        sim_control = SimulationControl()
        sim_control.sample_size = 2

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['process b'], [20.], rtol=.1)
        assert np.allclose(fp['use_phase_energy']['process a'], [24.], rtol=.1)

    def test_dict_deep_update(self):
        a = {}
        b = {'p': {'k': 'v'}}
        a.update(b)
        assert a['p']['k'] == 'v'

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_prototype_add_property(self):
        doc = u"""
    Processes:        
      - name: proto
        type: prototype
        metadata:
          device_type: User Device

      - name: a
        prototype: proto
        formula:
          text: |
           return a + b
        input_variables:
          - formula_name: a
            type: StaticVariable
            value: 6
        import_variables:
          - b        
      - name: b
        formula:
          text: |
            c = 2
            a = 2                
        export_variables:      
          c: b
          a: d 
        link_to: 
          - a
    Metadata:
      model_name: test
    """

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        sim_control.use_time_series = False

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['a'], [8.], rtol=.1)

        for node in s.process_graph.nodes_iter():
            if node.name == 'a':
                assert 'metadata' in node.__dict__
                assert node.metadata['device_type'] == 'User Device'

        # @todo review return values from DSL

    @unittest.skip("old DSL. New does not return values - need review")
    def test_prototype_overwrite_property(self):
        doc = u"""
    Processes:        
      - name: proto
        type: prototype
        metadata:
          device_type: User Device
        formula:
          text: |
           return a         
        input_variables:
          - formula_name: g
            type: StaticVariable
            value: 2
      - name: a
        prototype: proto
        formula:
          text: |
           return a + b
        input_variables:
          - formula_name: a
            type: StaticVariable
            value: 6
          - formula_name: d
            type: StaticVariable
            value: 7
        import_variables:
          - b       
        metadata:
          device_type: Router
          platform_name: DSL
          
      - name: b
        formula:
          text: |
            c = 2
            a = 2                
        export_variables:      
          c: b
          a: d 
        link_to: 
          - a
    Metadata:
      model_name: test
    """

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        sim_control.use_time_series = False

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        traces = s.collect_calculation_traces()
        _variable_values = traces['energy']

        assert np.allclose(fp['use_phase_energy']['a'], [8.], rtol=.1)

        for node in s.process_graph.nodes_iter():
            if node.name == 'a':
                assert 'metadata' in node.__dict__
                assert node.metadata['device_type'] == 'Router'
                assert node.metadata['platform_name'] == 'DSL'

                assert 'g' in node.input_variables
                assert 'd' in node.input_variables


class YamlLoader_v2(unittest.TestCase):
    def test_basic(self):
        doc = u"""
variant: 2
Processes:
- name: UD
  formula: |-
    energy = power * time
    data = time * bitrate
  tableVariables:
  - value: power
  - value: bitrate
  importVariables:
  - value: time
  staticVariables: []
  exportVariables:
  - value: data
  metadata: []
  ui_data:
    x: 259.0788432267884
    y: 140.07387747336378
  link_to:
  - Net
- name: Net
  formula: |-
    energy = data * intensity
  tableVariables:
  - value: intensity
  importVariables:
  - value: data
  staticVariables: []
  exportVariables: []
  metadata: []
  ui_data:
    x: 447.2464300083151
    y: 155.97068535868286
  link_to: []
- name: User
  formula: |
  tableVariables: []
  importVariables: []
  staticVariables:
  - name: time
    value: '5 minutes'
  exportVariables:
  - value: time
  metadata: []
  ui_data:
    x: 200.68045794750105
    y: 235.9945607673532
  link_to:
  - UD
Metadata:
  model_name: test
  table_file_name: tests/data/yaml_loader_test_data.xlsx
    """

        sim_control = SimulationControl()
        sim_control.sample_size = 1
        sim_control.sample_mean_value = True
        sim_control.use_time_series = True

        yaml_structure = YamlLoader.load_definitions(doc)

        s = YamlLoader.create_service(yaml_structure, sim_control)

        for node in s.process_graph.nodes_iter():
            if node.name in ['Net']:
                assert node.formulaModel.formula.text == 'energy = data * intensity'
                assert s.process_graph.out_edges(node) == []

            if node.name == 'UD':

                linked_nodes = set()
                for (_, p) in s.process_graph.out_edges(node):
                    linked_nodes.add(p.name)
                assert linked_nodes == {'Net'}

        fp = s.footprint(embodied=False, simulation_control=sim_control)
        print(fp)


if __name__ == '__main__':
    unittest.main()
