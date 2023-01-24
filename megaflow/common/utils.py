import os
from argparse import ArgumentParser
from datetime import datetime
import torch
# from model import *
# from dataset import *
from metrics import *


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def initialize_model(in_channel, out_channel, type, layers, num_filters):
    # initialize model based on type, layers, and num_filters provided
    raise NotImplementedError


def initialize_dataset(dataset, mode, dir, transform):
    raise NotImplementedError


def initialize_loss(loss_type):
    """
    Initialize loss function based on type provided
    Input:
        loss_type: string, type of loss function
    Output:
        loss_fn: loss function
    """
    if loss_type == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    elif loss_type == 'L1Loss':
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))
    return loss_fn


def initialize_metric(metric_type):
    """
    Initialize metric function based on type provided
    Input:
        metric_type: string, type of metric function
    Output:
        metric_fn: metric function
    """
    if metric_type == 'max_divergence':
        metric_fn = max_divergence 
    elif metric_type == 'norm_divergence':
        metric_fn = norm_divergence
    else:
        raise ValueError('Unknown metric type: {}'.format(metric_type))
    return metric_fn


def evaluate_model(model, dataloader, logger, iteration, loss_fn, eval_metric, device, mode, checkpoint=None):
    raise NotImplementedError
