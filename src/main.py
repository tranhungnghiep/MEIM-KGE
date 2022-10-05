#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hung-Nghiep Tran

import os
import argparse
import logging
import time

import json
import copy
import itertools
import multiprocessing as mp

import random
import numpy as np
import torch

from models import MEIM, MEI, DistMult, CP, CPh, SimplE, ComplEx, RotatE, Quaternion, W2V, W2Vh, Random
from experiments import Experiment
import utils

def get_parser():
    """
    Store all configurations with default values.
    """
    # ===============
    # Define default values
    # ===============
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--model', default='MEIM', type=str, help='the class name of the model for experiment')
    parser.add_argument('--device', default='cuda', type=str, help='cuda, cpu')

    parser.add_argument('--config_id', default='', type=str, help='name of an experiment group: "" (default single exp), cid000, cid001... (find configs from config.json, batch experiment)')
    parser.add_argument('--gpu', default=[0], type=int, nargs='+', help='list of gpu ids, will auto convert to single id for each run, e.g., 0 2 4: [0]')
    parser.add_argument('--num_cpu_threads', default=1, type=int, help='number of cpu threads for data processing: <not implemented>')
    parser.add_argument('--num_gpu_threads', default=1, type=int, help='number of gpu threads per each gpu, i.e., number of exp to run in parallel on each gpu in batch experiment: 1')
    parser.add_argument('--max_num_exp', default=-1, type=int, help='max number of exp to run for random search in a cid batch experiment: -1 (run all loaded configs)')

    parser.add_argument('--single_exp', default=1, type=int, help='single exp (normal logging) or multiple exps (only logging to file, no logging to console, hyperparams search): 1, 0')
    parser.add_argument('--debug', default=0, type=int, help='debug mode, turn on to print information to console when single_exp==0: 0, 1')
    parser.add_argument('--deterministic', default=0, type=int, help='for gpu deterministic mode; not important: 0, 1')
    parser.add_argument('--seed', default=7, type=int, help='for reproduce; used with random.seed(seed), np.random.seed(seed), tf.set_random_seed(seed), torch.manual_seed(seed): 7')
    parser.add_argument('--profiling', default=0, type=int, help='to print/log time profiling of each action: 0, 1')

    parser.add_argument('--logging', default=1, type=int, help='to print/log training progress or not: 1, 0')
    parser.add_argument('--check_period', default=5, type=int, help='how many epochs after which to check results on train/valid/test')
    parser.add_argument('--eval_scoreorder', default='mean', type=str, help='how to treat score order in ranking evaluation: mean, min, max (most papers, complex, cph, ... use min; generally good enough, mr+-.005, mrr+-5e-8, except pathological cases like ReLU output; we use mean, guaranteed)')
    parser.add_argument('--early_stop', default=0, type=int, help='to early stop by mrr on valid or not: 0, 1')
    parser.add_argument('--patience', default=10, type=int, help='how many times for patience before early stop')
    parser.add_argument('--eval_intype', default=0, type=int, help='to evaluate filtering by ent types or not, hard-coded types (currently only support bib kg, raise error otherwise): 0, 1')
    parser.add_argument('--print_each_rel', default=0, type=int, help='to print each rel result when print best eval result or not (currently always eval each rel because it is very cheap): 0')
    parser.add_argument('--eval_eachpartition', default=0, type=int, help='for ensemble analysis, to evaluate each local partition or not: 0, 1')
    parser.add_argument('--eval_eachpartition_epochs', default=[], type=int, nargs='*', help='list of epoch to eval_each_partition: [], "0", "10", "10 50 100 200 300 400 500"')
    parser.add_argument('--write_ranks', default=0, type=int, help='for ensemble analysis, to write ranks of each triple each direction or not: 0, 1')

    # Data and log
    parser.add_argument('--in_path', default='../datasets/wn18rr', type=str, help='general data dir: ../datasets/wn18rr, ../datasets/fb15k-237, etc.')
    parser.add_argument('--out_path', default='../result/temp', type=str, help='general output dir: ../result/temp')
    parser.add_argument('--tb_logdir', default='', type=str, help='tensorboard log dir: "", ../result/tb')
    parser.add_argument('--params_out_path', default="", type=str, help='semquery output dir: "", ../out')

    parser.add_argument('--exp_id', default='', type=str, help='optional prefix of experiment ID to log and save (auto postfix: time--model--dataset): ""')
    parser.add_argument('--save_model', default=0, type=int, help='to save model checkpoint or not, at export_step epochs/new best valid epoch/finish training: 0, 1')
    parser.add_argument('--export_step', default=0, type=int, help='export step to auto save model checkpoint: 0 (not auto saving)')
    parser.add_argument('--write_params', default=0, type=int, help='write params like embs and cores and mappings: 0 (not write), 1 (write at specifed epochs), >=20 (verbose auto write at new best valid)')
    parser.add_argument('--write_params_emb', default=0, type=int, help='write ent rel embs: 0, 1')
    parser.add_argument('--write_params_core', default=0, type=int, help='write cores: 0, 1')
    parser.add_argument('--write_params_mapping', default=0, type=int, help='write mapping matrices: 0, 1')
    parser.add_argument('--write_params_epochs', default=[], type=int, nargs='*', help='list of epoch to write emb: [], "0", "10", "10 50 100 200 300 400 500"')
    parser.add_argument('--write_params_burnin_epoch', default=0, type=int, help='minimum epoch for burnin before verbose auto writing params (write_params>=20): 0')

    # Embedding
    parser.add_argument('--K', default=3, type=int, help='number of partitions in MEI embedding')
    parser.add_argument('--Ce', default=100, type=int, help='ent emb components size: 1 for DistMult, 2 for CP/CPh/SimplE/ComplEx, 4 for Quaternion')
    parser.add_argument('--Cr', default=100, type=int, help='rel emb components size: 1 for DistMult, 2 for CP/CPh/SimplE/ComplEx, 4 for Quaternion')
    parser.add_argument('--core_tensor', default='nonshared', type=str, help='independent or dependent core tensor: nonshared, shared')
    parser.add_argument('--combine_score', default='sum', type=str, help='sum, mean')
    parser.add_argument('--compute_score', default='any', type=str, help='legacy: use default or specific compute_score method for shared and nonshared core tensor: any, specific')
    parser.add_argument('--init_emb', default=1, type=int, help='how to init emb: 1 (xavier: 2/(fanin+fanout)), 2 (1/fanout), 3 (1/c)')
    parser.add_argument('--init_w', default=0, type=int, help='how to init wv: 0 (rep tf truncnorm -1, 1, m0, std0.5), 1 (k*k), 2 (1/k), 3 (k), 4 (1)')
    parser.add_argument('--init_emb_gain', default=1e-2, type=float, help='gain scaling of init emb std: use init xavier and gain 1e-2 to reproduce tf xavier on 3-order tensor')

    # Optimization
    parser.add_argument('--reuse_array', default='torch1pin', type=str, help='reuse array for each batch or not (no change to computation and result): [torch|np][1|0][|pin|gpu]')
    parser.add_argument('--tail_fraction', default=1.0, type=float, help='fraction of entities to sample for the tail in kvsall and 1vsall: 1.0')
    parser.add_argument('--sampling', default='kvsall', type=str, help='which negative sampling method: kvsall (conve), 1vsall (lacroix), kvssome (conve), 1vssome (kadlec), negsamp (transe)')
    parser.add_argument('--model_inv', default=0, type=int, help='always use 2 way loss, either by model inverse or data inverse: 0 (by data: for easy implementation), 1 (inverse by model: 2 way model score and loss) <not implemented>')

    parser.add_argument('--loss_mode', default='softmax-cross-entropy', type=str, help='softmax-cross-entropy (categorical), cross-entropy (binary), softplus (binary), margin, mix-cross-entropy (binary and categorical), weightcorrected-softmax-cross-entropy (corrected kvsall sampling for softmax loss)')

    parser.add_argument('--margin', default=.5, type=float, help='for margin loss only; margin in transx model: .5')
    parser.add_argument('--binary_weight', default=.5, type=float, help='for mix loss only; weight of binary cross-entropy loss: .5')

    parser.add_argument('--batch_size', default=1024, type=int, help='training batch size: 1024')
    parser.add_argument('--batch_size_test', default=0, type=int, help='legacy in negsamp evaluation; for each test triple evaluate to all entities, this limits it to evaluate to batch_size_test entities, optional: 0 (evaluate to all entities)')
    parser.add_argument('--neg_ratio', default=1, type=int, help='legacy in negsamp sampling method; average negative ratio for each triple, note that head or tail are sampled randomly, totally neg_ratio times for each positive triple: 1')
    parser.add_argument('--max_epoch', default=1000, type=int, help='max training epochs: 1000')

    parser.add_argument('--opt_method', default='adam', type=str, help='optimizer: adam, adagrad, sgd...')
    parser.add_argument('--amsgrad', default=0, type=int, help='to use amsgrad variant of adam or not: 0, 1')
    parser.add_argument('--adagrad_iav', default=.01, type=float, help='initial_accumulator_value in adagrad: .01, 0.001')
    parser.add_argument('--lr', default=3e-3, type=float, help='learning rate: usually 1e-2 to 1e-3 for adam, 1e0 to 1e-1 for adagrad')
    parser.add_argument('--lr_scheduler', default='exp', type=str, help='learning rate scheduler method: "" (constant lr), exp (exponential), rdop (reduce on plateau), warmup_rdop (warmup lr a few epochs first, then rdop freely), warmup_exp (warmup lr then exponential decay): exp is good enough')
    parser.add_argument('--lr_decay', default=.99775, type=float, help='lr decay exponentially (lr_decayed = lr * lr_decay**epoch) (will implement cosine and reduce on plateau, only 3 main decays): 1 (no decay), .99775, .9975, .995...')
    parser.add_argument('--lr_minlr', default=5e-5, type=float, help='min lr: 5e-5, 0 (torch rdop)')
    parser.add_argument('--lr_rdop_f', default=.9, type=float, help='rdop factor (lr_decayed = lr * factor = lr * factor**times): .9, .1 (torch)')
    parser.add_argument('--lr_rdop_p', default=25, type=int, help='rdop patience before decay: 10 (torch)')
    parser.add_argument('--lr_rdop_t', default=1e-4, type=float, help='rdop threshold to say plateau, relative value: 1e-4 (torch)')
    parser.add_argument('--lr_rdop_watch', default='val_mrr', type=str, help='rdop metrics to watch plateau: val_mrr (mode min, remember to pass -mrr; it makes sense to use main metric mrr here, no need val_loss), val_loss (torch)')
    parser.add_argument('--lr_warmup_maxlr', default=1e-2, type=float, help='warmup max lr: 1e-2')
    parser.add_argument('--lr_warmup_epoch', default=40, type=int, help='warmup how many epochs: 40')
    parser.add_argument('--lr_warmup_space', default='lin', type=str, help='warmup step space: lin, log')

    # Regularization
    parser.add_argument('--lambda_reg', default=.0, type=float, help='weight decay reg in general')
    parser.add_argument('--lambda_ent', default=.0, type=float, help='weight decay reg for entity emb: .0, 1e-7 ... 1e0')
    parser.add_argument('--lambda_rel', default=.0, type=float, help='weight decay reg for relation emb: .0, -1 (shared with lambda_ent), 1e-7 ... 1e0')
    parser.add_argument('--lambda_params', default=.0, type=float, help='weight decay reg for other params: .0, 1e-7 ... 1e0')
    parser.add_argument('--reg_pnorm', default=3., type=float, help='weight decay reg pnorm: 3.0 for nuclear n3, 2.0 for frobenius l2, 1.0 for sparse l1')
    parser.add_argument('--reg_temp', default=1., type=float, help='temperature to increase or decrease unequal raw frequency; w = c**t/sum(c**t): temperature==1. (raw triple weight, default), temperature==0. (raw batch weight)')
    parser.add_argument('--reg_weightedsum', default=1, type=int, help='normalize unique weight to sum 1.0 or not: 1 (weighted sum, mean over batch), 0 (sum over batch)')
    parser.add_argument('--reg_decayedfactor', default='raw', type=str, help='raw: decay each emb entry, rownorm: decay emb row norm (with N3 for ComplEx)')

    parser.add_argument('--mapping_constraint', default='', type=str, help='mapping type or Mr: '' (no constraint), orthogonal...')
    parser.add_argument('--lambda_ortho', default=.0, type=float, help='weight of orthogonal constraint loss: .0, 1e-7 ... 1e0')
    parser.add_argument('--ortho_dim', default='col', type=str, help='which dimension to constrain orthogonality: col, row, both')
    parser.add_argument('--ortho_p', default=2., type=float, help='orthogonal pnorm, distance from identity matrix: 2.0 for frobenius l2, 3.0 for nuclear n3, 1.0 for sparse l1')
    parser.add_argument('--ortho_by_w', default=0, type=int, help='to compute separate mr using r.detach() so that ortho only depends on w or not: 0')
    parser.add_argument('--ortho_droprate_mr', default=.0, type=float, help='droprate for mr in orthogonal loss computation: .0')

    parser.add_argument('--lambda_rowrelnorm', default=.0, type=float, help='weight of rowrel norm loss to push rowrel norm in addition to push mr ortho: .0, 1e-7 ... 1e0, -1 (= lambda_ortho), -2 (0.1 lambda_ortho)')
    parser.add_argument('--rowrelnorm_c', default=1., type=float, help='rowrel norm distance from c: 1.0 for rowrel unitnorm, 0.0 for small rowrel norm; e.g., soft transx, soft rotate and soft quaternion use rowrelnorm_c==1.0 with large lambda_rowrelnorm')
    parser.add_argument('--rowrelnorm_p', default=3., type=float, help='rowrel pnorm: 3.0 for nuclear n3, 2.0 for frobenius l2, 1.0 for sparse l1')
    parser.add_argument('--rowrelnorm_droprate_r', default=.0, type=float, help='droprate for r in soft unitnorm rowrel loss computation: .0')

    parser.add_argument('--constraint', default='', type=str, help='constraint input entity emb: "", nonneg, unitnorm, minmaxnorm...; e.g., transx uses constraint==unitnorm and to_constrain==colent, rotate and quaternion use constraint==unitnorm and to_constrain==rowrel')
    parser.add_argument('--to_constrain', default='', type=str, help='which emb and dimension to constrain: "", rowent, colent, fullent..., rowrel, colrel, fullrel..., rowent_rowrel...; (note that here col is K dim, row is C dim)')

    parser.add_argument('--shift_score', default=0, type=int, help='score +y: 0')
    parser.add_argument('--scale_score', default=1, type=int, help='score *x: 1')

    parser.add_argument('--label_smooth_style', default='tensorflow', type=str, help='label smoothing style: "tensorflow" (y = (1-e)*y + e/2), "conve" (y = (1-e)*y + 1/num_ents)')
    parser.add_argument('--label_smooth', default=.0, type=float, help='label smoothing e to push raw labels towards uniform binary: .0')

    parser.add_argument('--droprate', default=.0, type=float, help='droprate for dropout in general')
    parser.add_argument('--droprate_w', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate_r', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate_mr', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate_h', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate_mrh', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate_t', default=.0, type=float, help='droprate for dropout in MEI')

    parser.add_argument('--norm', default='bn', type=str, help='bn (batchnorm), ln (layernorm)')

    parser.add_argument('--n', default=0, type=int, help='normalize or not in general: 0, 1')
    parser.add_argument('--n_w', default=0, type=int, help='normalize or not in MEI: 0, 1')
    parser.add_argument('--n_r', default=0, type=int, help='normalize or not in MEI: 0, 1')
    parser.add_argument('--n_mr', default=0, type=int, help='normalize or not in MEI: 0, 1')
    parser.add_argument('--n_h', default=0, type=int, help='normalize or not in MEI: 0, 1')
    parser.add_argument('--n_mrh', default=0, type=int, help='normalize or not in MEI: 0, 1')
    parser.add_argument('--n_t', default=0, type=int, help='normalize or not in MEI: 0, 1')

    parser.add_argument('--n_epsilon', default=1e-3, type=float, help='epsilon in batchnorm: 1e-3')
    parser.add_argument('--n_sepK', default=1, type=int, help='separate n statistics for K partitions or not in MEI: 1, 0')

    parser.add_argument('--bn_momentum', default=.01, type=float, help='momentum in batchnorm: .01 in torch == .99 in tf')
    parser.add_argument('--bn_renorm', default=0, type=int, help='legacy: tf batch renorm or not for noisy batch in MEI: 0, 1')

    # General
    parser.add_argument('--epsilon', default=1e-12, type=float, help='for numerical stable computation')
    parser.add_argument('--max_value', default=1e100, type=float, help='for numerical stable computation')

    return parser


def get_config(parser, arg_str=None):
    """
    Parse values from arg_str (if not None, default None) or command line (default).
    :return: Namespace dict, can be updated from code later.
    """
    if arg_str:
        config, unknown_args = parser.parse_known_args(arg_str.split())
    else:
        config, unknown_args = parser.parse_known_args()
    if unknown_args:
        print('\nWARNING!!! THERE ARE UNKNOWN ARGS IN ARG_STR: %s!!!\n' % str(unknown_args))

    config.argkey_orig = list(vars(config).keys())  # list() make a new copy of set, freeze order = declared order

    return config


def load_config(config, config_file='../config.json'):
    """
    z: 0; a: 1, 2; b: 3, 4 -> first z0a1b3; then z0a1b4, z0a2b3, z0a2b4
    Use itertools.product to make all combination of all hparams values
    Deep copy config and set new attributes values

    All changes are based on updating the deep copy of original 'config'.
    Return the same ['config'] if no change or no config generated from file.
    """
    config_list = []

    if config.config_id:
        with open(config_file, 'r') as f:
            hp = json.load(f)
        
        if config.config_id in hp:
            hp = hp[config.config_id]  # get to specific config's hparams dict

            k_list = []  # key list: [z, a, b]
            vs_list = []  # values list of each key:  [[0], [1, 2], [3, 4]]
            for k in hp.keys():
                if k in config.argkey_orig:
                    k_list.append(k)
                    vs_list.append(hp[k])
                else:
                    print('\nWARNING!!! THERE ARE UNKNOWN ARGS IN CONFIG_FILE: %s!!!\n' % str(k))

            for v_list in itertools.product(*vs_list):  # [[0,1,3], [0,1,4], [0,2,3], [0,2,4]]
                new_c = copy.deepcopy(config)
                for k, v in zip(k_list, v_list):
                    setattr(new_c, k, v)
                config_list.append(new_c)

    if len(config_list) == 0:
        config_list.append(copy.deepcopy(config))  # use copy of original config if no config generated from file

    for c in config_list:
        c.single_exp = 1 if len(config_list) == 1 else 0  # single exp if only one config, even from config.json
        c.exp_id = c.exp_id + config.config_id + '_' if config.config_id else c.exp_id  # add config_id to prefix of exp_id

    return config_list


def autoupdate_config(config):
    # supported model classes, map with config string by a dictionary
    model_dict = {
        'MEIM': MEIM,
        'MEI': MEI,
        'DistMult': DistMult,
        'CP': CP,
        'CPh': CPh,
        'SimplE': SimplE,
        'ComplEx': ComplEx,
        'RotatE': RotatE,
        'Quaternion': Quaternion,
        'W2V': W2V,
        'W2Vh': W2Vh,
        'Random': Random
    }
    config.model_class = model_dict[config.model]

    # auto compute machine friendly arguments based on human friendly arguments
    # axis(0, 1, 2) = (batch, k, c) = (batch, col, row)
    if 'rowent' in config.to_constrain:
        config.constrain_axis_ent = 2
    elif 'colent' in config.to_constrain:
        config.constrain_axis_ent = 1
    elif 'fullent' in config.to_constrain:
        config.constrain_axis_ent = (1, 2)
    else:
        config.constrain_axis_ent = 2

    if 'rowrel' in config.to_constrain:
        config.constrain_axis_rel = 2
    elif 'colrel' in config.to_constrain:
        config.constrain_axis_rel = 1
    elif 'fullrel' in config.to_constrain:
        config.constrain_axis_rel = (1, 2)
    else:
        config.constrain_axis_rel = 2

    # sharing lambda for ent, rel
    if config.lambda_rel == -1:
        print('\nWARNING!!! SHARING LAMBDA WITH ENT, REL: %s!!!' % str(config.lambda_ent))
        config.lambda_rel = config.lambda_ent

    # sharing lambda for ortho and rowrelnorm
    lambda_rowrelnorm_dict = {
        -1: 1,
        -2: 1e-1,
        -2.5: 5e-2,
        -3: 1e-2,
        -3.5: 5e-3,
        -4: 1e-3,
        -4.5: 5e-4
    }
    if config.lambda_rowrelnorm < 0:
        if config.lambda_rowrelnorm in lambda_rowrelnorm_dict:
            lambda_rowrelnorm_factor = lambda_rowrelnorm_dict[config.lambda_rowrelnorm]
        else:
            lambda_rowrelnorm_factor = -config.lambda_rowrelnorm
        print('\nWARNING!!! SHARING LAMBDA WITH ORTHO, ROWRELNORM: %s * %s!!!' % (str(lambda_rowrelnorm_factor), str(config.lambda_ortho)))
        config.lambda_rowrelnorm = lambda_rowrelnorm_factor * config.lambda_ortho

    # sharing droprate for h, mrh
    if config.droprate_mrh == -1:
        print('\nWARNING!!! SHARING DROPRATE WITH H, MRH: %s!!!' % str(config.droprate_h))
        config.droprate_mrh = config.droprate_h

    # sharing droprate for r, mr
    if config.droprate_mr == -1:
        print('\nWARNING!!! SHARING DROPRATE WITH R, MR: %s!!!' % str(config.droprate_r))
        config.droprate_mr = config.droprate_r

    # sharing droprate for rowrelnorm r, r
    if config.rowrelnorm_droprate_r == -1:
        print('\nWARNING!!! SHARING DROPRATE WITH R, ROWRELNORM R: %s!!!' % str(config.droprate_r))
        config.rowrelnorm_droprate_r = config.droprate_r

    if config.logging:
        # create necessary dirs
        if config.out_path and config.logging:
            os.makedirs(os.path.join(config.out_path, 'log'), exist_ok=True)
        # logger, in one process (main), getLogger with the same name returns the same logger, set for all config objects
        config.logger_debug = utils.get_logger('debug', loglevel=logging.DEBUG,
                                               file_paths=(os.path.join(config.out_path, 'log', 'debug.log'),), file_loglevel=logging.DEBUG,
                                               stream=True, stream_loglevel=logging.DEBUG)
        # filename <= 255 bytes, so we use unique exp_id based on timestamp as filename and keep track in a master file
        config.logger_master_finish = utils.get_logger('master_finish', loglevel=logging.DEBUG,
                                                       file_paths=
                                                       (os.path.join(config.out_path, 'log', 'master_finish.log'),
                                                        os.path.join(config.out_path, 'log', 'master_finish_%s.log' % config.config_id),) if config.config_id
                                                       else (os.path.join(config.out_path, 'log', 'master_finish.log'),),
                                                       file_loglevel=logging.INFO)

    return config


def main():
    # ===============
    # original config
    # ===============
    print('Loading config\n')
    parser = get_parser()
    config = get_config(parser)
    config = autoupdate_config(config)

    # ===============
    # make all configs to parallel massive search hparams,
    # these configs can be then sampled by grid/random/bayesian,
    # call exp in subprocess with selected config
    # ===============
    config_list = load_config(config, config_file='../config.json')
    config_list = list(map(autoupdate_config, config_list))

    # random search
    time_seed = time.time_ns() % (2**32 - 1)
    random.seed(time_seed)
    np.random.seed(time_seed)
    np.random.shuffle(config_list)  # random exp order
    num_configs_total = len(config_list)
    if config.max_num_exp == -1:
        config.max_num_exp = num_configs_total
    config_list = config_list[:config.max_num_exp]
    num_configs_sampled = len(config_list)

    # assign gpu: use all gpu, then repeat num_gpu_threads time, e.g., 012 012 ...; run pool num gpus * num_gpu_threads
    for i, c in enumerate(config_list):
        c.gpu = config.gpu[i % len(config.gpu)]

    # ===============
    # run all configs
    # ===============
    config.logger_debug.info('\nSTART%s\n' % ' in %s' % config.config_id if config.config_id else "")
    config.logger_debug.info('There are %i experiments (%.00f%% of %i) in this %srun\n'
                             % (num_configs_sampled, np.round(num_configs_sampled/num_configs_total*100), num_configs_total,
                                '%s ' % config.config_id if config.config_id else ""))

    lock = mp.Lock()
    with mp.Pool(len(config.gpu) * config.num_gpu_threads, initializer=pool_init, initargs=(lock,)) as pool:  # old: control number of threads per gpu with each gpu separately; new: control group of gpu together
        pool.map(run_exp, config_list)

    config.logger_debug.info('FINISH%s\n' % ' in %s' % config.config_id if config.config_id else "")


def pool_init(lock):
    global pool_lock
    pool_lock = lock


def run_exp(config):
    config.logger_debug.info('Starting a new experiment%s\n' % ' in %s' % config.config_id if config.config_id else "")

    # ===============
    # experiment, a single experiment using a single config, can run in a subprocess
    # ===============
    exp = Experiment(config=config)
    exp.run()

    if config.logging:
        # logger is not process safe when writing to a shared file, depends on luck, or use a lock
        with pool_lock:
            config.logger_master_finish.info('%s\t%s\tTRAIN\t%s\tVALID\t%s\tTEST\t%s\t%i\t%E\t%.3f' %
                                             (exp.config.exp_id, exp.config.arg_str,
                                              '%s\t%s' % (exp.format_eval_result_tuple(exp.best_eval_train['raw']), exp.format_eval_result_tuple(exp.best_eval_train['filter'])),
                                              '%s\t%s' % (exp.format_eval_result_tuple(exp.best_eval_valid['raw']), exp.format_eval_result_tuple(exp.best_eval_valid['filter'])),
                                              '%s\t%s' % (exp.format_eval_result_tuple(exp.best_eval_test['raw']), exp.format_eval_result_tuple(exp.best_eval_test['filter'])),
                                              exp.best_epoch,
                                              exp.best_epoch_loss,
                                              exp.best_eval_valid['filter'][1]))

    # ===============
    # test, optional sanity test
    # ===============
    if config.debug:
        with pool_lock:
            if 'wn18' in config.in_path:  # Sanity test on wn18
                exp.show_link_prediction(h='06845599', t='03754979', r='_member_of_domain_usage', raw=True)
            if 'fb15k' in config.in_path:  # Sanity test on fb15k
                exp.show_link_prediction(h='/m/08966', t='/m/05lf_', r='/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month', raw=True)
            if 'KG20C' in config.in_path:  # Sanity test
                exp.show_link_prediction(h='794DCC81', t='7E7C0518', r='author_write_paper', raw=True)


if __name__ == '__main__':
    main()
