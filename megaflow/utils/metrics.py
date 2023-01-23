import numpy as np
import torch
import torch.nn.functional as F


def max_divergence(y_pred, y_true):
    """
    Computes the maximum divergence between the predicted and true distributions
    Input:
        y_pred: tensor, predicted distribution
        y_true: tensor, true distribution
    Output:
        max_div: float, maximum divergence between the predicted and true distributions
    """
    max_div = 1 - torch.max(torch.abs(y_pred - y_true)) / torch.max(y_true)
    return max_div


def norm_divergence(y_pred, y_true):
    """
    Computes the norm divergence between the predicted and true distributions
    Input:
        y_pred: tensor, predicted distribution
        y_true: tensor, true distribution
    Output:
        norm_div: float, norm divergence between the predicted and true distributions
    """
    norm_div = 1 - (torch.norm(y_pred) - torch.norm(y_true)) / torch.norm(y_true)
    return norm_div