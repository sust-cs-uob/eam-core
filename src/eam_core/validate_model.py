import datetime
import os.path
import re

import yaml
from ruamel import yaml

from table_data_reader.table_handlers import OpenpyxlTableHandler

import logging

logger = logging.getLogger(__name__)

statuses = ['development', 'testing', 'live', 'archived', 'test_resource']


class YAMLValidationError(ValueError):
    pass


def get_validated_statuses(level):
    validated_statuses = []
    level_reached = False
    for status in statuses:
        if status == level:
            level_reached = True

        if level_reached:
            validated_statuses.append(status)

    return validated_statuses


def get_status(yaml_struct):
    if 'Metadata' not in yaml_struct:
        raise YAMLValidationError('Metadata section missing from model yaml')

    if 'status' not in yaml_struct['Metadata']:
        raise YAMLValidationError('status missing from Metadata')

    if yaml_struct['Metadata']['status'] not in statuses:
        raise YAMLValidationError(f'status {yaml_struct["Metadata"]["status"]} not a valid status.\n'
                                  f'must be one of {statuses}')

    return yaml_struct['Metadata']['status']


def get_table_definitions(yaml_dir, yaml_struct):
    if 'table_file_name' not in yaml_struct['Metadata']:
        raise YAMLValidationError('table_file_name is missing from Metadata')

    if not isinstance(yaml_struct['Metadata']['table_file_name'], str):
        raise YAMLValidationError('table_file_name must be a string')

    table_file_name = f'{yaml_dir}/{yaml_struct["Metadata"]["table_file_name"]}'
    if not os.path.exists(table_file_name):
        raise YAMLValidationError('table_file_name must point to a file')

    # return OpenpyxlTableHandler().load_definitions(None, filename=table_file_name)
    return None


def validate_model(filepath, level='testing'):
    logger.info(f'Validating {filepath}')

    with open(filepath) as model:
        yaml_struct = yaml.load(model, Loader=yaml.RoundTripLoader)

        validated_statuses = get_validated_statuses(level)

        if get_status(yaml_struct) in validated_statuses:
            assert_not_forced_invalid(yaml_struct)

            yaml_dir = os.path.dirname(filepath)
            table_definitions = get_table_definitions(yaml_dir, yaml_struct)

            validate_metadata(yaml_struct['Metadata'])
        else:
            logger.info(f'Skipped: yaml has status {get_status(yaml_struct)}')


def assert_not_forced_invalid(yaml_struct):
    if yaml_struct.get('INVALID', False):
        raise YAMLValidationError('yaml forced to be invalid')


def validate_metadata(metadata):
    assert_required_metadata_present(metadata)
    assert_metadata_types_correct(metadata)
    assert_model_version_follows_semantic_versioning(metadata)

    if metadata.get('with_group', False):
        validate_countrified_metadata(metadata)


def validate_processes(yaml_struct):
    if 'Processes' not in yaml_struct:
        raise YAMLValidationError('Processes section missing from model yaml')


def assert_required_metadata_present(metadata):
    if 'model_name' not in metadata:
        raise YAMLValidationError('model_name is missing from Metadata')

    if 'model_version' not in metadata:
        raise YAMLValidationError('model_version is missing from Metadata')

    if 'start_date' not in metadata:
        raise YAMLValidationError('start_date is missing from Metadata')

    if 'end_date' not in metadata:
        raise YAMLValidationError('end_date is missing from Metadata')

    if 'sample_size' not in metadata:
        raise YAMLValidationError('sample_size is missing from Metadata')

    if 'sample_mean' not in metadata:
        raise YAMLValidationError('sample_mean is missing from Metadata')

    if 'description' not in metadata:
        logger.warning('description is missing from Metadata')

    if 'process_map' not in metadata:
        logger.warning('process_map is missing from Metadata')


def assert_metadata_types_correct(metadata):
    if not isinstance(metadata['model_name'], str):
        raise YAMLValidationError('model_name must be a string')

    if not isinstance(metadata['model_version'], str):
        raise YAMLValidationError('model_version must be a string')

    if not isinstance(metadata['start_date'], datetime.date):
        raise YAMLValidationError('start_date must be a date')

    if not isinstance(metadata['end_date'], datetime.date):
        raise YAMLValidationError('end_date must be a date')

    if not isinstance(metadata['sample_size'], int):
        raise YAMLValidationError('sample_size must be an integer')

    if not isinstance(metadata['sample_mean'], bool):
        raise YAMLValidationError('sample_mean must be a boolean')

    if 'description' in metadata:
        if not isinstance(metadata['description'], str):
            raise YAMLValidationError('description must be a string')

    if 'with_group' in metadata:
        if not isinstance(metadata['with_group'], bool):
            raise YAMLValidationError('with_group must be a boolean')


def assert_model_version_follows_semantic_versioning(metadata):
    if re.fullmatch('[0-9]*\.[0-9]*\.[0-9]', metadata['model_version']) is None:
        raise YAMLValidationError('model_version must follow Semantic Versioning')


def validate_countrified_metadata(metadata):
    assert_dependency_vars_are_countrified(metadata)
    assert_required_countrified_metadata_present(metadata)
    assert_countrified_metadata_types_correct(metadata)
    assert_countrified_vars_are_ui(metadata)


def assert_dependency_vars_are_countrified(metadata):
    if 'group_dependencies' in metadata:
        group_dependencies = metadata['group_dependencies']
        for group_dependency in group_dependencies:
            linked_param_names = list(group_dependency.values())[0]
            for linked_param_name in linked_param_names:
                if linked_param_name not in metadata['group_vars']:
                    raise YAMLValidationError(f'{linked_param_name} is in a group dependency, but is not in group_vars')


def assert_required_countrified_metadata_present(metadata):
    if 'group_vars' not in metadata:
        raise YAMLValidationError('with_group is true, but group_vars is missing from Metadata')

    """
    @todo unsure if aggregation is required group metadata, since its not always used.
    if 'group_aggregation_vars' not in metadata:
        raise YAMLValidationError('with_group is true, but group_aggregation_vars is missing from Metadata')
    """

def assert_countrified_metadata_types_correct(metadata):
    if not isinstance(metadata['group_vars'], list):
        raise YAMLValidationError('group_vars must be a list')

    """
    same issue as in assert_required_countrified_metadata_present()
    if not isinstance(metadata['group_aggregation_vars'], list):
        raise YAMLValidationError('group_aggregation_vars must be a list')
    """

    if 'groupings' in metadata:
        if not isinstance(metadata['groupings'], list):
            raise YAMLValidationError('groupings must be a list')


def assert_countrified_vars_are_ui(metadata):
    pass
