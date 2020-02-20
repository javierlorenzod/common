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
import random as rn
import os
import pickle as pkl

import numpy as np
import torch
import yaml

from config import argparser

logger = logging.getLogger(__name__)


def load_values_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            return nested_dict_to_object(config), config
    except yaml.YAMLError as exc:
        if hasattr(exc, 'problem_mark'):
            mark = exc.problem_mark
            print(f"Error position: ({mark.line + 1}:{mark.column + 1})")


# https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object?page=1&tab=votes#tab-top
class nested_dict_to_object(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [nested_dict_to_object(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, nested_dict_to_object(b) if isinstance(b, dict) else b)


def set_random_seeds(random_seed):
    np.random.seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    rn.seed(random_seed)

def check_configuration_arguments(args, kwds):
    cfg_file_path = None
    if len(args) == 1 and isinstance(args[0], str):
        cfg_file_path = args[0]
    elif len(args) == 0: # Case for KIT-MRT work --> redo
        args = argparser.parse_args()
        cfg_file_path = args.cfg_file_path
    if cfg_file_path is None:
        raise AttributeError('The configuration file path must be valid')
    else:
        logger.info(f"Loading config from file: {cfg_file_path}")
    return cfg_file_path
# TODO: Save configuration file copying it instead of loading it to a file [DONE]
def load_configuration(*args, **kwds):
    cfg_file_path = check_configuration_arguments(args, kwds)
    conf, confyaml = load_values_from_file(cfg_file_path)
    set_random_seeds(conf.Training.random_seed)
    return conf, confyaml, cfg_file_path

# TODO: Create a function for the common part of this group of configuration loading
def load_configuration_jaad(*args, **kwds):
    logger.info("Loading configuration specific of JAAD")
    cfg_file_path = check_configuration_arguments(args, kwds)
    conf, confyaml = load_values_from_file(cfg_file_path)
    return conf, confyaml, cfg_file_path

# TODO: delete this method (is the old one for RNNs and autoencoders)
# def load_configuration():
#     args = argparser.parse_args()
#     if args.cfg_file_path is None:
#         raise AttributeError('The configuration file path must be valid')
#     else:
#         conf, confyaml = load_values_from_file(args.cfg_file_path)
#     logger.info(f"Device used: {conf.Training.device}")
#     config_utils.set_random_seeds(conf.Training.random_seed)
#     return conf, confyaml


def save_config_for_model(save_path, config, input_cfg_path):
    filename, file_extension = os.path.splitext(save_path)
    with open(f"{filename}.pkl", 'wb') as outf:
        pkl.dump(config, outf, protocol=pkl.HIGHEST_PROTOCOL)
    with open(save_path, 'w') as outfile:
        yaml.safe_dump(config, outfile)
    with open(input_cfg_path, "r") as f:
        lines = f.readlines()
        with open(save_path, "w") as f1:
            f1.writelines(lines)

def save_config_for_model_iv20(save_path, config, input_cfg_path):
    filename, file_extension = os.path.splitext(save_path)
    with open(f"{filename}.pkl", 'wb') as outf:
        pkl.dump(config, outf, protocol=pkl.HIGHEST_PROTOCOL)
    with open(input_cfg_path, "r") as f:
        lines = f.readlines()
        with open(save_path, "w") as f1:
            f1.writelines(lines)

def get_cnn_extractor_model_name(conf):
    m_f_id = conf.Model.CNN.chosen_model_family # The ID of the chosen family
    n_families = len(conf.Model.CNN.model_families)
    chosen_family = conf.Model.CNN.model_families[m_f_id]
    c_s_id = conf.Model.CNN.chosen_submodel
    if chosen_family == "resnet":
        # Check if chosen submodel ID is valid
        n_submodels = len(conf.Model.CNN.model_variants_list[m_f_id])
        chosen_submodel = conf.Model.CNN.model_variants_list[m_f_id][c_s_id]
        cnn_extractor_model_name = f"{chosen_family}{chosen_submodel}"
        cnn_extractor_input_size = 224 # square input
    elif chosen_family == "convae_resnet":
        # Check if chosen submodel ID is valid
        n_submodels = len(conf.Model.CNN.model_variants_list[m_f_id])
        chosen_submodel = conf.Model.CNN.model_variants_list[m_f_id][c_s_id]
        cnn_extractor_model_name = f"{chosen_family}{chosen_submodel}"
        cnn_extractor_input_size = 224  # square input
    elif chosen_family == "convae2":
        # Check if chosen submodel ID is valid
        n_submodels = len(conf.Model.CNN.model_variants_list[m_f_id])
        chosen_submodel = conf.Model.CNN.model_variants_list[m_f_id][c_s_id]
        cnn_extractor_model_name = f"{chosen_family}{chosen_submodel}"
        cnn_extractor_input_size = 224  # square input
    return cnn_extractor_model_name