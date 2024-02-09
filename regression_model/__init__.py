import logging

from regression_model.config.core import PACKAGE_ROOT, config

with open(PACKAGE_ROOT/'VERSION') as version_file:
    __version__ = version_file.read().strip()


logging.getLogger(config.app_config.package_name).\
    addHandler(logging.NullHandler())
# print(__version__)
