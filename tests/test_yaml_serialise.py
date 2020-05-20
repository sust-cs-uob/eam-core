import json
import os
import unittest
from datetime import datetime, date
from typing import Dict

import numpy as np
import ruamel.yaml as yaml

from eam_core import SimulationControl, Variable
from eam_core.YamlLoader import YamlLoader
from ruamel.yaml.scalarstring import PreservedScalarString as pss

'''Create an encoder subclassing JSON.encoder.
Make this encoder aware of our classes (e.g. datetime.datetime objects)
'''


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime) or isinstance(obj, date):
            return obj.isoformat()
        else:
            return json.JSONEncoder.default(self, obj)


class MyTestCase(unittest.TestCase):

    @unittest.skip('cannot use teapot project reference here')  # @todo
    def test_scenarios(self):

        yaml_struct = None
        with open('../teapot_scenarios/models/scenarios.yml', 'r') as stream:
            try:
                yaml_struct = yaml.load(stream)
                # yaml_struct = yaml.load(stream, Loader=yaml.RoundTripLoader)
            except yaml.YAMLError as exc:
                print(exc)
                # logger.error(f'Error while loading yaml file {yamlfile} {exc}')
                # sys.exit(1)

        # print(yaml_struct)

        with open('from_yaml_scenarios.json', 'w') as outfile:
            json.dump(yaml_struct, outfile, cls=Encoder, indent=4)

        with open('from_yaml_scenarios.json', 'r') as jsonfile:
            _json = json.load(jsonfile)

        for proceses in _json['Processes']:
            if 'formula' in proceses:
                proceses['formula']['text'] = pss(proceses['formula']['text'])

        with open('from_yaml_scenarios.json.yml', 'w') as outfile:

            # yaml.dump(_json, outfile, default_flow_style=False, width=4096)
            yaml.dump(_json, outfile, default_flow_style=False, Dumper=yaml.RoundTripDumper, width=4096)

        # with open('scenarios.yml', 'w') as outfile:
        #     yaml.dump(yaml_struct, outfile, default_flow_style=False, Dumper=yaml.RoundTripDumper)

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

        if not os.path.exists('.tmp-test'):
            os.mkdir('.tmp-test')

        with open('.tmp-test/from_ymal.json', 'w') as outfile:
            json.dump(yaml_structure, outfile)

        s = YamlLoader.create_service(yaml_structure, sim_control)
        fp = s.footprint(embodied=False, simulation_control=sim_control)
        # print(fp)
        assert np.allclose(fp['use_phase_energy']['a'], [8.], rtol=.1)

        for node in s.process_graph.nodes_iter():
            if node.name == 'a':
                assert 'metadata' in node.__dict__
                assert node.metadata['device_type'] == 'User Device'

    def test_model_to_ymal(self):
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
            c = 2;
            a = 2;
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
        with open('from_ymal.json', 'w') as outfile:
            json.dump(yaml_structure, outfile)

        print(yaml.dump_all([s]))


if __name__ == '__main__':
    unittest.main()
