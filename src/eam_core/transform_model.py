import logging
from logging.config import dictConfig

from pip._vendor.pkg_resources import resource_filename, Requirement
from ruamel import yaml
from ruamel.yaml.scalarstring import PreservedScalarString as pss

import eam_core.log_configuration as logconf

logger = logging.getLogger(__name__)


def setup_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', help="input yaml file")
    parser.add_argument('--output-file', '-o', help="input yaml file")
    parser.add_argument('--verbose', '-v', help="enable debug level logging", action='store_true')
    args = parser.parse_args()
    # print(args)
    return args


def transform(infile_name, outfile_name):
    logger.info(f'reading {infile_name}')

    yaml_struct = None
    with open(infile_name, 'r') as stream:
        try:
            yaml_struct = yaml.load(stream)
            # yaml_struct = yaml.load(stream, Loader=yaml.RoundTripLoader)
        except yaml.YAMLError as exc:
            print(exc)

    _json = {'variant': 2,
             'Processes': [],
             'Metadata': yaml_struct['Metadata'], 'Analysis': yaml_struct['Analysis'],
             'Constants': []

             }

    # Gets list of prototypes
    prototypes = {}
    for proc in yaml_struct['Processes']:
        if 'type' in proc and proc['type'] == 'prototype':
            prototypes[proc['name']] = proc

    # Moves constants to new file
    if 'Constants' in yaml_struct:
        for proc in yaml_struct['Constants']:
            for k, v in proc['variables'].items():
                _json['Constants'].append({'name': k, 'value': v})

    for proc in yaml_struct['Processes']:
        # Ignore if process is a prototype
        if 'type' in proc and proc['type'] == 'prototype':
            continue

        # Create a process object
        logger.debug(f"{proc['name']}")
        prc_obj = {'name': proc['name'], 'metadata': proc.get('metadata', []),
                   'tableVariables': [],
                   'constants': [],
                   'importVariables': [],
                   'exportVariables': [],
                   }

        # If process has a prototype copy prototype into process
        if 'prototype' in proc:
            prtype = prototypes[proc['prototype']]
            for k, v in prtype.items():
                if k == 'name':     # Keep process' original name
                    continue
                prc_obj[k] = v

        # Copy any link_to into new process
        if 'link_to' in proc:
            prc_obj['link_to'] = proc['link_to']

        # Copy any formulas into new process
        if 'formula' in proc:
            prc_obj['formula'] = pss(proc['formula']['text'])

        # Copy input variables
        for ivar_section in proc.get('input_variables', []):
            if ivar_section['type'] == 'ExcelVariableSet':
                for ivar in ivar_section['variables']:
                    prc_obj['tableVariables'].append({'value': ivar})

            if ivar_section['type'] == 'Constants':
                for ivar in ivar_section['variables']:
                    prc_obj['constants'].append({'value': ivar})

        # Copy import variables
        for import_var in proc.get('import_variables', []):
            prc_obj['importVariables'].append({'value': import_var})

        # Copy export variables
        for export_var in proc.get('export_variables', []):
            prc_obj['exportVariables'].append({'value': export_var})

        logger.debug(prc_obj)
        _json['Processes'].append(prc_obj)

    with open(outfile_name, 'w') as outfile:
        logger.info(f'writing {outfile_name}')
        # yaml.dump(_json, outfile, default_flow_style=False, width=4096)
        noalias_dumper = yaml.RoundTripDumper
        noalias_dumper.ignore_aliases = lambda self, data: True
        # print yaml.dump(data, default_flow_style=False, Dumper=noalias_dumper)
        yaml.dump(_json, outfile, default_flow_style=False, Dumper=noalias_dumper, width=4096)


if __name__ == '__main__':
    args = setup_parser()
    logger.info(f"Running with parameters {args}")
    if args.verbose:
        level = logging.DEBUG
        logger = logging.getLogger('ngmodel')
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    transform(args.input_file, args.output_file)
