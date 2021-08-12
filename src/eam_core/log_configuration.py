import logging.handlers

from logging.config import dictConfig

from pip._vendor.pkg_resources import resource_filename, Requirement
from ruamel import yaml
import os

configured = False


def config_logging():
    global configured
    if configured:
        logging.info(f"Logging already configured.")
        return
    configured = True

    log_conf_file = os.getenv('LOG_CONFIG')  # None

    if not log_conf_file:
        log_conf_file = resource_filename(Requirement.parse('eam_core'), "logconf.yml")
        if not os.path.exists(log_conf_file):
            import pathlib
            log_conf_file = pathlib.Path(__file__).parent.absolute().joinpath("logconf.yml")

    with open(log_conf_file, 'r') as f:
        log_config = yaml.safe_load(f.read())
    dictConfig(log_config)
    logging.info(f"Configured logging from {log_conf_file}")


class MakeFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.handlers.RotatingFileHandler.__init__(self, filename, mode, maxBytes, backupCount, encoding, delay)


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


# todo: unused. delete?
def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': WHITE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, use_color=True):
        logging.Formatter.__init__(self,
                                   "[\033[1m%(name)-20s\033[0m][%(levelname)-18s]  %(message)s (\033[1m%(filename)s\033[0m:%(lineno)d)")
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            lvl_bkp = record.levelname
            record.levelname = levelname_color
            formatter_format = logging.Formatter.format(self, record)
            record.levelname = lvl_bkp
        return formatter_format
