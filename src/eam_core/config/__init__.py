import os
from jinja2 import Environment, FileSystemLoader


def render_template(template_name, template_directory):
    tmpl_env = Environment(loader=FileSystemLoader(template_directory))
    template = tmpl_env.get_template(template_name)
    return template.render(env=os.environ)


def write_templated_file(target_file, template_file, template_directory):
    with open(target_file, "w") as fh:
        fh.write(render_template(template_file, template_directory))


def create_project_config():
    assert 'GOOGLE_DRIVE_DATA_SHEET_IDS' in os.environ
    write_templated_file('local.cfg', 'local.cfg.template', 'etc/templates')
