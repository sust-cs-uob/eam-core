
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

    return yaml_struct['Metadata']['status']


def validate_model(yaml_struct, level='testing'):
    validated_statuses = get_validated_statuses(level)

    if get_status(yaml_struct) in validated_statuses:
        pass


def assert_countrified_vars_are_ui():
    pass
