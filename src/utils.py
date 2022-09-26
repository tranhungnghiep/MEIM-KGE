import os
import time
import datetime

import random
import numpy as np
import scipy as sp
import torch

import logging


def to_torch(a: np.ndarray, device) -> torch.Tensor:
    if device == 'cuda':
        return torch.as_tensor(a).cuda()
    else:
        return torch.as_tensor(a)


def to_numpy(b: torch.Tensor) -> np.ndarray:
    if b.device.type == 'cuda':
        return b.cpu().detach().numpy()
    else:
        return b.detach().numpy()


def truncated_normal_(tensor: torch.Tensor, myclip_a=-1, myclip_b=1, my_mean=0.0, my_std=0.5):
    """
    Fills an empty tensor values from a truncated normal distribution, based on scipy
    Note that scipy take a, b range in standard normal domain, need to convert from my range to that range
    :param tensor: empty tensor
    :param myclip_a: expected output range start
    :param myclip_b: expected output range end
    :param my_mean: expected output mean
    :param my_std: expected output std
    :return: filled tensor
    """
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std  # a, b == -2, 2
    from scipy import stats as spstats
    with torch.no_grad():
        # Tensor.data to update the underlying tensor
        tensor.data[:] = to_torch(spstats.truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=list(tensor.shape)), tensor.device.type)
    return tensor


def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def metrics(ranks):
    """
    Main metrics for the link prediction task.
    :param ranks: (n, ) 1-d numpy array containing ranks of all triples (and both directions by inv)
    :return: MR, MRR, H@1, H@3, H@10
    """
    if len(ranks) > 0:
        mr, mrr, h1, h3, h10 = np.mean(ranks), np.mean(1.0 / ranks), np.mean(ranks <= 1), np.mean(ranks <= 3), np.mean(ranks <= 10)
    else:
        mr, mrr, h1, h3, h10 = np.inf, 0.0, 0.0, 0.0, 0.0  # if not available ranks, assign worst results
    return [mr, mrr, h1, h3, h10]


def nonneg(x):
    x = torch.clamp_min(x, 0)
    return x


def unitnorm(x, dim, p=2.0, epsilon=1e-12):
    x = x / (torch.norm(x, p=p, dim=dim, keepdim=True) + epsilon)
    return x


def minmaxnorm(x, dim, p=2.0, min_norm=0.0, max_norm=2.0, epsilon=1e-12):
    x_norm = torch.norm(x, p=p, dim=dim, keepdim=True)
    x_norm_new = torch.clamp(x_norm, min_norm, max_norm)
    x = x / (x_norm + epsilon) * x_norm_new
    return x


def get_logger(name, loglevel=logging.DEBUG,
               file_paths=(), file_loglevel=logging.INFO,
               stream=False, stream_loglevel=logging.DEBUG):
    logger = logging.getLogger(name)  # need distinct name
    logger.handlers.clear()  # reset handlers, to be not duplicate
    logger.setLevel(loglevel)  # default level of this logger
    if file_paths:
        for file_path in file_paths:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(file_loglevel)  # write from info level up to file
            logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_loglevel)  # write every level to console, for temporary debugging
        logger.addHandler(stream_handler)

    return logger


def get_tb_writer(log_dir='./tb'):
    import tensorflow as tf
    import tensorboard as tb
    if tb.__version__.startswith('2'):
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # fix bug on tb/tf2.0: AttributeError: module 'tensorflow_core._api.v2.io.gfile' has no attribute 'get_filesystem'
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=log_dir)
