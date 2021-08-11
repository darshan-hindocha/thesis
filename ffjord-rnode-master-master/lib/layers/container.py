import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=True, inds=None,density=False):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1) # if len(self.chain) is 10, then inds = 9,8,...,1,0
            else:
                inds = range(len(self.chain))
        if density:
            for i in inds:
                x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse,density=density)
            return x, logpx, reg_states
        
        for i in inds:
            x = self.chain[i](x, logpx, reg_states, reverse=reverse,density=density)
        return x, None, None
