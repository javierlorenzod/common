#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'

import logging
from loguru import logger

def setup_custom_logger(name):
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

# TODO: Meter opciones para subrayar y poner en negrita, cursiva etc
def log_colorful(message: str, color: str) -> None:
    logger.opt(ansi=True).info(f"<{color}>{message}</{color}>")