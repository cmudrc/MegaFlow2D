import os
from argparse import ArgumentParser
from datetime import datetime
import torch
from model import *
from dataset import *
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
    if type == 'FlowMLConvolution':
        model = FlowMLConvolution(in_channel, out_channel, layers, num_filters)
    elif type == 'FlowMLError':
        model = FlowMLError(in_channel, out_channel)
    else:
        raise ValueError('Unknown model type: {}'.format(type))
    return model


def initialize_dataset(dataset, mode, dir, transform):
    # initialize dataset based on dataset and mode
    if dataset == 'MegaFlow2D':
        dataset = MegaFlow2D(root=dir, mode=mode, transform=transform)
        if dataset.is_processed is False:
            dataset.process()
        else:
            print('Dataset initialized')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    return dataset


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
    # load checkpoint if provided
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        avg_metric = 0

        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(batch.y, pred)
            metric = eval_metric(batch.y, pred)
            avg_loss += loss.item()
            avg_metric += metric

        avg_loss /= len(dataloader)
        avg_metric /= len(dataloader)

        if mode == 'val':
            logger.add_scalar('Loss/val', avg_loss, iteration)
            logger.add_scalar('Max_div/val', avg_metric, iteration)
            print('-' * 72)
            print('Val loss: {:.4f}, Val metric: {:.4f}'.format(avg_loss, avg_metric))

        if mode == 'test':
            logger.add_scalar('test_loss', avg_loss, iteration)
            logger.add_scalar('test_metric', avg_metric, iteration)
            print('-' * 72)
            print('Test loss: {:.4f}, Test metric: {:.4f}'.format(avg_loss, avg_metric))

    model.train()
    return avg_loss, avg_metric


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MegaFlow2D', help='dataset name')
    parser.add_argument('--mode', type=str, default='mixed', help='dataset mode')
    parser.add_argument('--transform', type=str, default='None', help='dataset transform')
    parser.add_argument('--dir', type=str, default='C:/research/data', help='dataset directory')
    parser.add_argument('--model', type=str, default='FlowMLConvolution', help='model type')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--num_filters', type=int, nargs='+', default=[8, 16, 8], help='number of filters')
    parser.add_argument('--loss', type=str, default='MSELoss', help='loss function')
    parser.add_argument('--metric', type=str, default='max_divergence', help='metric function')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--load_model', type=str, default=None, help='load model from checkpoint')

    args = parser.parse_args()
    return args
