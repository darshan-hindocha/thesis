import os
import math
from numbers import Number
import logging
import torch
import six
from core_lib.layers.wrappers import cnf_regularization as reg_lib
from core_lib import layers
from core_lib.layers.odefunc import divergence_bf, divergence_approx


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def set_step_size(step_size, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver_options['step_size'] = step_size

    model.apply(_set)


def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size
            if args.first_step is not None:
                module.solver_options['first_step'] = args.first_step

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol
            if args.test_step_size is not None:
                module.test_solver_options['step_size'] = args.test_step_size
            if args.test_first_step is not None:
                module.test_solver_options['first_step'] = args.test_first_step


    model.apply(_set)


def override_divergence_fn(model, divergence_fn):

    def _set(module):
        if isinstance(module, layers.ODEfunc):
            if divergence_fn == "brute_force":
                module.divergence_fn = divergence_bf
            elif divergence_fn == "approximate":
                module.divergence_fn = divergence_approx

    model.apply(_set)


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def append_regularization_to_log(log_message, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = log_message + " | " + INV_REGULARIZATION_FNS[reg_fn] + ": {:.2e}".format(reg_states[i].item())
    return log_message


def append_regularization_keys_header(header, regularization_fns):
    for reg_fn in regularization_fns:
        header.append(INV_REGULARIZATION_FNS[reg_fn])
    return header


def append_regularization_csv_dict(d, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        d[INV_REGULARIZATION_FNS[reg_fn]] = '{:.4f}'.format(reg_states[i].item())
    return d


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))

    for module in model.modules():
        if isinstance(module, layers.CNF):
            reg = module.get_regularization_states()
            acc_reg_states = tuple(acc_reg_states[i] + reg[i] for i in range(len(reg)))

    return acc_reg_states
