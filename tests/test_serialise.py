import json
import unittest

import yaml

from eam_core import FormulaModel, FormulaProcess, SimulationControl, ServiceModel, generate_static_variable, Formula
from eam_core import complex_encoder
from eam_core.YamlLoader import YamlLoader


class MyTestCase(unittest.TestCase):

    @unittest.skip('Has no assertion, doesn\'t seem to do anything useful')
    def test_to_json(self):
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
                c = 3;
                b = 2;
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
            # fp = s.footprint(embodied=False, simulation_control=sim_control)

            json_str: str = json.JSONEncoder(default=complex_encoder).encode(s)
            # print(json_str)

    @unittest.skip('Has no assertion, doesn\'t seem to do anything useful')
    def test_serialise_to_yaml(self):
        sim_control = SimulationControl()
        sim_control.sample_size = 1

        s = ServiceModel()

        input_variables = {
            'data_volume_var': generate_static_variable(sim_control, 'data_volume_var', 5, random=False)}
        dvpm = FormulaModel(Formula("data_volume = data_volume_var;"), )
        # todo - define return variables
        export_variables = {'data_volume': 'data_volume'}
        dvp = FormulaProcess('test', dvpm, input_variables=input_variables, export_variable_names=export_variables)

        test_formula = """
                energy = energy_intensity * data_volume;
                """
        input_variables = {
            'energy_intensity': generate_static_variable(sim_control, 'energy_intensity', 5, random=False)}

        fmodel = FormulaModel(Formula(test_formula), )

        import_variables = {'data_volume': {'aggregate': True}}
        p = FormulaProcess('test', fmodel, input_variables=input_variables, import_variable_names=import_variables)

        # print(yaml.dump_all([fmodel, p]))


if __name__ == '__main__':
    unittest.main()
