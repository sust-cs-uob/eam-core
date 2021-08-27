import datetime
import unittest
from typing import Dict

import numpy as np
import yaml
from ruamel import yaml

from eam_core import SimulationControl, Variable
from eam_core.YamlLoader import YamlLoader
from tests.directory_test_controller import use_test_dir

from eam_core.validate_model import validate_model, YAMLValidationError


def load_valid_model():
    with use_test_dir():
        with open('models/valid.yml') as model:
            return yaml.load(model, Loader=yaml.RoundTripLoader)


def load_valid_group_model():
    with use_test_dir():
        with open('models/valid_group.yml') as model:
            return yaml.load(model, Loader=yaml.RoundTripLoader)


class TestLevels(unittest.TestCase):

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


class ValidationTestCase(unittest.TestCase):

    def assert_validate_throws_error(self, model, message):
        with self.assertRaises(Exception) as context:
            validate_model(model)

        assert str(context.exception) == message, \
            f'Expected exception message to be \'{message}\', but was \'{str(context.exception)}\''


class TestValid(unittest.TestCase):

    def test_valid_model(self):
        model = load_valid_model()

        try:
            validate_model(model)
        except YAMLValidationError as e:
            raise AssertionError(f'Expected no errors, but \'{str(e)}\' was raised.')

    def test_description_added(self):
        model = load_valid_model()
        model['Metadata']['Description'] = 'test'

        try:
            validate_model(model)
        except YAMLValidationError as e:
            raise AssertionError(f'Expected no errors, but \'{str(e)}\' was raised.')


class TestMissing(ValidationTestCase):

    def test_missing_model_name(self):
        model = load_valid_model()
        del model['Metadata']['model_name']
        self.assert_validate_throws_error(model, 'model_name is missing from Metadata')

    def test_missing_model_version(self):
        model = load_valid_model()
        del model['Metadata']['model_version']
        self.assert_validate_throws_error(model, 'model_version is missing from Metadata')

    def test_missing_table_file_name(self):
        model = load_valid_model()
        del model['Metadata']['table_file_name']
        self.assert_validate_throws_error(model, 'table_file_name is missing from Metadata')

    def test_missing_start_date(self):
        model = load_valid_model()
        del model['Metadata']['start_date']
        self.assert_validate_throws_error(model, 'start_date is missing from Metadata')

    def test_missing_end_date(self):
        model = load_valid_model()
        del model['Metadata']['end_date']
        self.assert_validate_throws_error(model, 'end_date is missing from Metadata')

    def test_missing_sample_size(self):
        model = load_valid_model()
        del model['Metadata']['sample_size']
        self.assert_validate_throws_error(model, 'sample_size is missing from Metadata')

    def test_missing_sample_mean(self):
        model = load_valid_model()
        del model['Metadata']['sample_mean']
        self.assert_validate_throws_error(model, 'sample_mean is missing from Metadata')


class TestMissingGroup(ValidationTestCase):

    def test_missing_group_vars(self):
        model = load_valid_group_model()
        del model['Metadata']['group_vars']
        self.assert_validate_throws_error(model, 'with_group is true, but group_vars is missing '
                                                                    'from Metadata')

    def test_missing_group_aggregation_vars(self):
        model = load_valid_group_model()
        del model['Metadata']['group_aggregation_vars']
        self.assert_validate_throws_error(model, 'with_group is true, but group_aggregation_vars is '
                                                                    'missing from Metadata')


class TestTypes(ValidationTestCase):

    def test_model_name_not_string(self):
        model = load_valid_model()
        model['Metadata']['model_name'] = 1
        self.assert_validate_throws_error(model, 'model_name must be a string')

    def test_model_version_not_string(self):
        model = load_valid_model()
        model['Metadata']['model_version'] = []
        self.assert_validate_throws_error(model, 'model_version must be a string')

    def test_table_file_name_not_string(self):
        model = load_valid_model()
        model['Metadata']['table_file_name'] = False
        self.assert_validate_throws_error(model, 'table_file_name must be a string')

    def test_description_not_string(self):
        model = load_valid_model()
        model['Metadata']['description'] = {}
        self.assert_validate_throws_error(model, 'description must be a string')

    def test_start_date_not_date(self):
        model = load_valid_model()
        model['Metadata']['start_date'] = 'aaa'
        self.assert_validate_throws_error(model, 'start_date must be a date')

    def test_end_date_not_date(self):
        model = load_valid_model()
        model['Metadata']['end_date'] = 65E-9
        self.assert_validate_throws_error(model, 'end_date must be a date')

    def test_sample_size_not_integer(self):
        model = load_valid_model()
        model['Metadata']['sample_size'] = 0.5
        self.assert_validate_throws_error(model, 'sample_size must be an integer')

    def test_sample_mean_not_bool(self):
        model = load_valid_model()
        model['Metadata']['sample_mean'] = 3
        self.assert_validate_throws_error(model, 'sample_mean must be a boolean')


class TestTypesGroup(ValidationTestCase):

    def test_with_group_not_bool(self):
        model = load_valid_group_model()
        model['Metadata']['with_group'] = 8
        self.assert_validate_throws_error(model, 'with_group must be a boolean')

    def test_group_vars_not_list(self):
        model = load_valid_group_model()
        model['Metadata']['group_vars'] = 'eee'
        self.assert_validate_throws_error(model, 'group_vars must be a list')

    def test_group_aggregation_vars_not_list(self):
        model = load_valid_group_model()
        model['Metadata']['group_aggregation_vars'] = {'banana': 12}
        self.assert_validate_throws_error(model, 'group_aggregation_vars must be a list')

    def test_groupings_not_list(self):
        model = load_valid_group_model()
        model['Metadata']['groupings'] = datetime.date(2021, 1, 1)
        self.assert_validate_throws_error(model, 'groupings must be a list')


class TestInvalidValues(ValidationTestCase):

    def test_model_version_not_semantic_versioning(self):
        model = load_valid_model()


if __name__ == '__main__':
    unittest.main()
