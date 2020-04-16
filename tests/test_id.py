import eam_core.util as util
import unittest
import os
from eam_core.yaml_runner import load_configuration, setup_parser
from eam_core.YamlLoader import YamlLoader
from types import SimpleNamespace

class MyTestCase(unittest.TestCase):

    def test_existing_ids(self):
        dict = {"IDs" : True , "analysis_config" : "dev" , "comment" : None, "docker" : True , "documentation" : False , "filetype" : "pdf" , "local" : True , "sensitivity" : False , "verbose" : False , "yamlfile" : "tests/models/existing_ids.yml"}
        args = SimpleNamespace(**dict)
        # args = setup_parser("-l -D -d -id -a dev tests/models/ci_v2.yml")
        model_run_base_directory, simulation_run_description, yaml_struct = load_configuration(args)
        scenario = "default"
        model_output_directory = model_run_base_directory + f"/{scenario}"
        if not os.path.exists(model_output_directory):
            os.makedirs(model_output_directory)
        create_model_func, sim_control, yaml_struct = util.prepare_simulation(model_output_directory,
                                                                              simulation_run_description, yaml_struct,
                                                                              scenario, filename=args.yamlfile,
                                                                              IDs=args.IDs)
        loader = YamlLoader(version=yaml_struct.get('variant', 1))
        loader.parse_yaml_structure(yaml_struct, sim_control)
        assert loader.id_map["process"] == { "CDN" : 0 , "Internet Network" : 1 , "Laptop" : 2}