import datetime
import unittest
from typing import Dict

import numpy as np
import yaml

from eam_core import SimulationControl, Variable
from eam_core.YamlLoader import YamlLoader
from tests.directory_test_controller import use_test_dir

from eam_core.validate_model import validate_model, YAMLValidationError


class TestValidateModel(unittest.TestCase):

    def load_valid_model(self):
        with use_test_dir():
            with open('models/valid.yml') as model:
                return yaml.load(model), None

    def assert_invalid_throws_with_level(self, level, status):
        with self.assertRaises(Exception) as context:
            validate_model({
                'Metadata': {
                    'status': status
                },
                'INVALID': True
            }, None, level)

        assert str(context.exception) == 'yaml forced to be invalid',\
            f'Expected exception message to be \'yaml forced to be invalid\', but was \'{str(context.exception)}\''

    def assert_invalid_doesnt_throw_with_level(self, level, status):
        validate_model({
            'Metadata': {
                'status': status
            },
            'INVALID': True
        }, None, level)

    def test_level_development(self):
        self.assert_invalid_throws_with_level('development', 'development')
        self.assert_invalid_throws_with_level('development', 'testing')
        self.assert_invalid_throws_with_level('development', 'live')
        self.assert_invalid_throws_with_level('development', 'archived')
        self.assert_invalid_throws_with_level('development', 'test_resource')

    def test_level_testing(self):
        self.assert_invalid_doesnt_throw_with_level('testing', 'development')
        self.assert_invalid_throws_with_level('testing', 'testing')
        self.assert_invalid_throws_with_level('testing', 'live')
        self.assert_invalid_throws_with_level('testing', 'archived')
        self.assert_invalid_throws_with_level('testing', 'test_resource')

    def test_level_live(self):
        self.assert_invalid_doesnt_throw_with_level('live', 'development')
        self.assert_invalid_doesnt_throw_with_level('live', 'testing')
        self.assert_invalid_throws_with_level('live', 'live')
        self.assert_invalid_throws_with_level('live', 'archived')
        self.assert_invalid_throws_with_level('live', 'test_resource')

    def test_level_archived(self):
        self.assert_invalid_doesnt_throw_with_level('archived', 'development')
        self.assert_invalid_doesnt_throw_with_level('archived', 'testing')
        self.assert_invalid_doesnt_throw_with_level('archived', 'live')
        self.assert_invalid_throws_with_level('archived', 'archived')
        self.assert_invalid_throws_with_level('archived', 'test_resource')

    def test_level_test_resource(self):
        self.assert_invalid_doesnt_throw_with_level('test_resource', 'development')
        self.assert_invalid_doesnt_throw_with_level('test_resource', 'testing')
        self.assert_invalid_doesnt_throw_with_level('test_resource', 'live')
        self.assert_invalid_doesnt_throw_with_level('test_resource', 'archived')
        self.assert_invalid_throws_with_level('test_resource', 'test_resource')

    def test_valid_model(self):
        model, table_definitions = self.load_valid_model()

        try:
            validate_model(model, table_definitions)
        except YAMLValidationError as e:
            raise AssertionError(f'Expected no errors, but \'{str(e)}\' was raised.')

    def test_missing_model_name(self):
        model, table_definitions = self.load_valid_model()
        del model['Metadata']['model_name']

        with self.assertRaises(Exception) as context:
            validate_model(model, None)

        assert str(context.exception) == 'model_name is missing from Metadata',\
            f'Expected exception message to be \'model_name is missing from Metadata\', but was '\
            f'\'{str(context.exception)}\''

    def test_missing_model_version(self):
        model, table_definitions = self.load_valid_model()
        del model['Metadata']['model_version']

        with self.assertRaises(Exception) as context:
            validate_model(model, None)

        assert str(context.exception) == 'model_version is missing from Metadata',\
            f'Expected exception message to be \'model_version is missing from Metadata\', but was '\
            f'\'{str(context.exception)}\''


if __name__ == '__main__':
    unittest.main()
