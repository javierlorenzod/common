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

import datetime
from loguru import logger
from time import time, strftime, gmtime


def timing(f):
    def wrap(*args):
        time1 = time()
        ret = f(*args)
        time2 = time()
        logger.info('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))

        return ret

    return wrap


def get_current_time_str():
    now = datetime.datetime.now()
    return now.strftime("%d%m%y_%H%M%S")

def get_current_time_str_itsc2020():
    now = datetime.datetime.now()
    return now.strftime("%y%m%d_%H%M%S")