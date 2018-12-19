#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

import numpy as np
import os.path as ops
import re

from utils.arg_pars import opt


def estimate_prior(video_list):
    if ops.exists(ops.join(opt.dataset_root, opt.prior)):
        return
    prior = np.zeros((513, ))
    video_length = {}
    video_action_sets = {}
    with open(ops.join(opt.dataset_root, 'common_descriptors.txt'), 'r') as f:
        for line in f:
            search = re.search(r'(.*.txt)\s*(\d*)([\s*\d*]*)', line)
            video_name = search.group(1)
            n_frames = search.group(2)
            video_length[video_name] = int(n_frames)
            action_set = search.group(3).strip().split()
            action_set = [int(i) for i in action_set]
            video_action_sets[video_name] = action_set
    for video_name in video_list:
        video_actions = video_action_sets[video_name]
        n_frames = video_length[video_name]
        for action in video_actions:
            prior[action] += n_frames

    prior = prior / np.sum(prior)
    with open(ops.join(opt.dataset_root, opt.prior), 'w') as f:
        for single_prior in prior:
            f.write('%f\n' % single_prior)


def estimate_length_mean(video_list):
    if ops.exists(ops.join(opt.dataset_root, opt.length_mean)):
        return
    mean_length = np.zeros((513, ))
    action_counter = np.zeros((513, ))
    video_length = {}
    video_action_sets = {}
    with open(ops.join(opt.dataset_root, 'common_descriptors.txt'), 'r') as f:
        for line in f:
            search = re.search(r'(.*.txt)\s*(\d*)([\s*\d*]*)', line)
            video_name = search.group(1)
            n_frames = search.group(2)
            video_length[video_name] = int(n_frames)
            action_set = search.group(3).strip().split()
            action_set = [int(i) for i in action_set]
            video_action_sets[video_name] = action_set
    for video_name in video_list:
        video_actions = video_action_sets[video_name]
        n_frames = video_length[video_name]
        for action in video_actions:
            mean_length[action] += n_frames
            action_counter[action] += 1

    mean_length /= action_counter
    mean_length = np.nan_to_num(mean_length)
    with open(ops.join(opt.dataset_root, opt.length_mean), 'w') as f:
        for single_mean in mean_length:
            f.write('%f\n' % single_mean)


# ### prior ######################################################################
# def estimate_prior(dataset):
#     prior = np.zeros((dataset.n_classes,), dtype=np.float32)
#     for video in dataset.videos:
#         for c in range(dataset.n_classes):
#             prior[c] += video.n_frames if c in video.action_set else 0
#     return prior / np.sum(prior)


### loss based lengths #########################################################
def loss_based_lengths(dataset):
    # definition of objective function
    def objective(x, A, l):
        return 0.5 * np.sum((np.dot(A, x) - l) ** 2)

    # number of frames per video
    # vid_lengths = np.array([dataset.length(video) for video in dataset.videos()])
    # binary data matrix: (n_videos x n_classes), A[video, class] = 1 iff class in action_set[video]
    A = np.zeros((len(dataset), dataset.n_classes))
    for video_idx, video in enumerate(dataset.videos):
        for c in video.action_set:
            A[video_idx, c] = 1
    # constraints: each mean length is at least 50 frames
    constr = [lambda x, i=video_idx: x[i] - 50 for i in range(dataset.n_classes)]
    # optimize
    x0 = np.ones((dataset.n_classes)) * 450.0  # some initial value
    # mean_lengths = scipy.optimize.fmin_cobyla(objective, x0, constr, args=(A, vid_lengths), consargs=(), maxfun=10000, disp=False)
    mean_lengths = x0
    return mean_lengths


### monte-carlo grammar ########################################################
def monte_carlo_grammar(dataset, mean_lengths, index2label, max_paths=1000):
    monte_carlo_grammar = []
    sil_length = mean_lengths[0]
    while len(monte_carlo_grammar) < max_paths:
        for video in dataset.videos:
            action_set = video.action_set - set([0])  # exclude SIL
            seq = []
            while sum([mean_lengths[label] for label in seq]) + 2 * sil_length < dataset.length(video):
                seq.append(np.random.choice(list(action_set)))
            if len(seq) == 0:  # omit empty sequences
                continue
            monte_carlo_grammar.append('SIL ' + ' '.join([index2label[idx] for idx in seq]) + ' SIL')
    np.random.shuffle(monte_carlo_grammar)
    return monte_carlo_grammar[0:max_paths]