
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


def validate_model(yaml_struct, table_definitions, level='testing'):
    validated_statuses = get_validated_statuses(level)

    if get_status(yaml_struct) in validated_statuses:
        assert_not_forced_invalid(yaml_struct)

        validate_metadata(yaml_struct, table_definitions)


def assert_not_forced_invalid(yaml_struct):
    if yaml_struct.get('INVALID', False):
        raise YAMLValidationError('yaml forced to be invalid')


def validate_metadata(yaml_struct, table_definitions):
    assert_required_metadata_present(yaml_struct)

    if yaml_struct['Metadata'].get('with_group', False):
        validate_countrified_metadata(yaml_struct, table_definitions)


def validate_processes(yaml_struct, table_definitions):
    if 'Processes' not in yaml_struct:
        raise YAMLValidationError('Processes section missing from model yaml')


def assert_required_metadata_present(yaml_struct):
    if 'model_name' not in yaml_struct['Metadata']:
        raise YAMLValidationError('model_name is missing from Metadata')

    if 'model_version' not in yaml_struct['Metadata']:
        raise YAMLValidationError('model_version is missing from Metadata')

    if 'table_file_name' not in yaml_struct['Metadata']:
        raise YAMLValidationError('table_file_name is missing from Metadata')


def validate_countrified_metadata(yaml_struct, table_definitions):
    assert_required_countrified_metadata_present(yaml_struct)
    assert_countrified_vars_are_ui(yaml_struct, table_definitions)


def assert_required_countrified_metadata_present(yaml_struct):
    if 'group_vars' not in yaml_struct['Metadata']:
        raise YAMLValidationError('with_group is true, but group_vars is missing from Metadata')

    if 'group_aggregation_vars' not in yaml_struct['Metadata']:
        raise YAMLValidationError('with_group is true, but group_aggregation_vars is missing from Metadata')


def assert_countrified_vars_are_ui(yaml_struct, table_definitions):
    pass
