import math
import torch
import torch.nn as nn

_DEFAULT_ALPHA = 1e-6


class ZeroMeanTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.odelayer=False

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            x = x + .5
            if logpx is None:
                return x
            return x, logpx, reg_states
        else:
            x = x - .5
            if logpx is None:
                return x
            return x, logpx, reg_states


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.odelayer=False

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            out = _sigmoid(x, logpx, self.alpha)
            return out[0], out[1], reg_states
        else:
            out = _logit(x, logpx, self.alpha)
            return out[0], out[1], reg_states


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.odelayer=False

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            out = _logit(x, logpx, self.alpha)
            return out[0], out[1], reg_states
        else:
            out = _sigmoid(x, logpx, self.alpha)
            return out[0], out[1], reg_states


def _logit(x, logpx=None, alpha=_DEFAULT_ALPHA):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if logpx is None:
        return y
    return y, logpx - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _sigmoid(y, logpy=None, alpha=_DEFAULT_ALPHA):
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if logpy is None:
        return x
    return x, logpy + _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad
