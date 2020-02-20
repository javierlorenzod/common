#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions used to obtain metrics of any kind related to dataset and physical
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


def distance_from_point_to_line(point, line):
    """
    Calculate the distance from point to line using the formula in https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    :param point: list of 2 elements: [xp, yp]
    :param line: list of 2 point elements: [xl1, yl1, xl2, yl2]
    :return:
    """
    x0, y0 = point
    x1, y1, x2, y2 = line
    return abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1) / ((y2-y1)**2+(x2-x1)**2)**0.5

if __name__ == "__main__":
    point = [3.1, 2.2]
    line = [2.64, 5.92, 6.74, 0.86]
    line2 = [0.76, 3.3, 5.42, 5.88]
    print(distance_from_point_to_line(point, line2))