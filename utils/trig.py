"""
Copied from /home/jld/git-repos/VRU-detection-classification/pedestrian_action_recognition/simple_labeler/trig.py but modified
"""
import math
# TODO: Order functions and create a general functions module used by several projects
from enum import Enum, unique


@unique
class Bb(Enum):
    """
    Enumeration in order to help in the position of each piece of data inside a bounding box element
    """
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3


def calculate_intersection(bb1, bb2):
    w_bb1 = bb1[Bb.xmax.value] - bb1[Bb.xmin.value]
    h_bb2 = bb2[Bb.ymax.value] - bb2[Bb.ymin.value]
    w_bb2 = bb2[Bb.xmax.value] - bb2[Bb.xmin.value]
    h_bb1 = bb1[Bb.ymax.value] - bb1[Bb.ymin.value]
    w_intersection = min(w_bb1 + bb1[Bb.xmin.value], w_bb2 + bb2[Bb.xmin.value]) - max(bb1[Bb.xmin.value],
                                                                                       bb2[Bb.xmin.value])
    h_intersection = min(h_bb1 + bb1[Bb.ymin.value], h_bb2 + bb2[Bb.ymin.value]) - max(bb1[Bb.ymin.value],
                                                                                       bb2[Bb.ymin.value])
    if w_intersection <= 0 or h_intersection <= 0:
        area_intersection = 0
    else:
        area_intersection = w_intersection * h_intersection
    return area_intersection


def calculate_area(bb):
    h_bb = bb[Bb.ymax.value] - bb[Bb.ymin.value]
    w_bb = bb[Bb.xmax.value] - bb[Bb.xmin.value]
    area = h_bb * w_bb
    return area

def check_if_zero(val):
    return math.isclose(val, 0, rel_tol=1e-09)

def calculate_miou(bb1, bb2):
    """
    Modified Intersection over Union (mIoU)
    Intersection / minimum bounding box area
    """
    miou = 0
    bbs_intersection = calculate_intersection(bb1, bb2)
    if check_if_zero(bbs_intersection) is False:
        area1 = calculate_area(bb1)
        area2 = calculate_area(bb2)
        miou = bbs_intersection / min(area1, area2)
    return miou

def calculate_iou(bb1, bb2):
    """
    Modified Intersection over Union (mIoU)
    Intersection / minimum bounding box area
    """
    iou = 0
    bbs_intersection = calculate_intersection(bb1, bb2)
    if check_if_zero(bbs_intersection) is False:
        area1 = calculate_area(bb1)
        area2 = calculate_area(bb2)
        iou = bbs_intersection / (area1 + area2)
    return iou