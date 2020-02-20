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

import matplotlib.pyplot as plt
import numpy as np
import random as rn
import os
import torch.nn as nn

plt.style.use('ggplot')


def draw_results(sequences_length, input_sequences, output_sequences, predicted_horizon, predicted_sequences, vis_path,
                 epoch_id, visualization_type, test_name, visualization_title):
    if visualization_type is 'sequence':
        fig, axs = plt.subplots(len(sequences_length), 1, figsize=(5, len(sequences_length)), sharey=True)
        for test_seq_id in range(len(sequences_length)):
            axs[test_seq_id].plot(np.arange(input_sequences.size(0)),
                                  input_sequences[:, test_seq_id, 0].cpu().numpy(), 'g',
                                  linewidth=2.0)
            axs[test_seq_id].plot(
                np.arange(predicted_horizon, output_sequences.size(0) + predicted_horizon),
                output_sequences[:, test_seq_id, 0].cpu().numpy(), 'r', linewidth=2.0)
            axs[test_seq_id].plot(
                np.arange(predicted_horizon, predicted_sequences.size(0) + predicted_horizon),
                predicted_sequences[:, test_seq_id, 0].cpu().numpy(), 'b' + ':', linewidth=2.0)
        plt.savefig(os.path.join(vis_path, f'predict{epoch_id:04d}.jpg'))
        plt.close()
    elif visualization_type == 'trajectory':
        fig, axs = plt.subplots(len(sequences_length), 1, figsize=(5, len(sequences_length)), sharey=True, sharex=True)
        for test_seq_id in range(len(sequences_length)):
            axs[test_seq_id].plot(input_sequences[0:sequences_length[test_seq_id], test_seq_id, 0].cpu().numpy(),
                                     input_sequences[0:sequences_length[test_seq_id], test_seq_id, 1].cpu().numpy(), c='g', linewidth=1.0, linestyle='-')
            axs[test_seq_id].plot(output_sequences[0:sequences_length[test_seq_id], test_seq_id, 0].cpu().numpy(),
                                     output_sequences[0:sequences_length[test_seq_id], test_seq_id, 1].cpu().numpy(), c='r', linewidth=1.0, linestyle='-')
            axs[test_seq_id].plot(predicted_sequences[0:sequences_length[test_seq_id], test_seq_id, 0].cpu().numpy(),
                                     predicted_sequences[0:sequences_length[test_seq_id], test_seq_id, 1].cpu().numpy(), c='b', linewidth=1.0, linestyle='--')
        plt.savefig(os.path.join(vis_path, f'predict{epoch_id:04d}.jpg'))
        plt.close()
    elif visualization_type == 'trajectory_ae':
        sequences_plotted = rn.sample(range(input_sequences.shape[1]), 10)
        fig, axs = plt.subplots(10, 1, figsize=(5, 10), sharey=True)
        fig.suptitle(visualization_title)
        for test_seq_id, real_test_seq_id in enumerate(sequences_plotted):
            # axs[test_seq_id].set_xlabel('timesteps [1 ts = 1/16 s]')
            # axs[test_seq_id].set_ylabel('distance to the curb [m]')
            axs[test_seq_id].plot(input_sequences[:, real_test_seq_id, 0].cpu().numpy(),
                                     input_sequences[:, real_test_seq_id, 1].cpu().numpy(), c='g', linewidth=1.0, linestyle='-')
            axs[test_seq_id].plot(output_sequences[:, real_test_seq_id, 0].cpu().numpy(),
                                     output_sequences[:, real_test_seq_id, 1].cpu().numpy(), c='r', linewidth=1.0, linestyle='-')
            axs[test_seq_id].plot(predicted_sequences[:, real_test_seq_id, 0].cpu().numpy(),
                                     predicted_sequences[:, real_test_seq_id, 1].cpu().numpy(), c='b', linewidth=1.0, linestyle='--')
        plt.savefig(os.path.join(vis_path, f'predict{epoch_id:04d}.jpg'))
        plt.close()
    elif visualization_type is 'intention':
        predicted_sequences = nn.functional.softmax(predicted_sequences, 2)
        if len(sequences_length) > 10:
            n_sequences = 10
        else:
            n_sequences = len(sequences_length)
        sequences_chosen = rn.sample(range(len(sequences_length)), n_sequences)
        fig, axs = plt.subplots(n_sequences, 1, figsize=(5, 10), sharey=True)
        fig.suptitle(visualization_title)
        for test_seq_id, real_test_seq_id in enumerate(sequences_chosen):
            # axs[test_seq_id].set_xlabel('timesteps [1 ts = 1/16 s]')
            # axs[test_seq_id].set_ylabel('distance to the curb [m]')
            axs[test_seq_id].plot(np.arange(input_sequences.size(0)),
                                  input_sequences[:, real_test_seq_id, 0].cpu().numpy(), 'g',
                                  linewidth=2.0)
            axs[test_seq_id].fill_between(
                np.arange(predicted_horizon, output_sequences.size(0) + predicted_horizon),
                (output_sequences[:, real_test_seq_id, 0] * 10).cpu().numpy(), facecolor='red', alpha=0.25)
            axs[test_seq_id].fill_between(
                np.arange(predicted_horizon, output_sequences.size(0) + predicted_horizon),
                (output_sequences[:, real_test_seq_id, 1] * 10).cpu().numpy(), facecolor='cyan', alpha=0.25)
            axs[test_seq_id].fill_between(
                np.arange(predicted_horizon, predicted_sequences.size(0) + predicted_horizon),
                (predicted_sequences[:, real_test_seq_id, 0] * 10).cpu().numpy(), facecolor='blue', alpha=0.75)
            axs[test_seq_id].fill_between(
                np.arange(predicted_horizon, predicted_sequences.size(0) + predicted_horizon),
                (predicted_sequences[:, real_test_seq_id, 1] * 10).cpu().numpy(), facecolor='yellow', alpha=0.75)
        plt.savefig(os.path.join(vis_path, f'predict{epoch_id:04d}.jpg'))
        plt.close()
    elif visualization_type is 'vaesubseqs':
        # select randomly a group of 10 subsequences
        predicted_sequences = nn.functional.softmax(predicted_sequences, 2)
        sequences_plotted = rn.sample(range(input_sequences.shape[1]), 10)
        fig, axs = plt.subplots(10, 2, figsize=(5, 10), sharex=True)
        fig.suptitle(visualization_title)
        for testid, test_seq_id in enumerate(sequences_plotted):
            axs[testid][0].plot(np.arange(input_sequences.size(0)),
                                input_sequences[:, test_seq_id, 0].cpu().numpy(), 'g',
                                linewidth=2.0)
            axs[testid][1].fill_between(
                np.arange(input_sequences.size(0), input_sequences.size(0) + output_sequences.size(0)),
                (output_sequences[:, test_seq_id, 0] * 10).cpu().numpy(), facecolor='red', alpha=0.25)
            axs[testid][1].fill_between(
                np.arange(input_sequences.size(0), input_sequences.size(0) + output_sequences.size(0)),
                (output_sequences[:, test_seq_id, 1] * 10).cpu().numpy(), facecolor='cyan', alpha=0.25)
            axs[testid][1].fill_between(
                np.arange(input_sequences.size(0), input_sequences.size(0) + output_sequences.size(0)),
                (predicted_sequences[:, test_seq_id, 0] * 10).cpu().numpy(), facecolor='blue', alpha=0.25)
            axs[testid][1].fill_between(
                np.arange(input_sequences.size(0), input_sequences.size(0) + output_sequences.size(0)),
                (predicted_sequences[:, test_seq_id, 1] * 10).cpu().numpy(), facecolor='yellow', alpha=0.25)
        plt.savefig(os.path.join(vis_path, f'{test_name}_predict{epoch_id:04d}.jpg'))
        plt.close()
