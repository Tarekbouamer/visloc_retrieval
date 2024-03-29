from collections import OrderedDict, defaultdict
from math import log10

import tensorboardX as tensorboard
import torch
import torchvision.transforms as transforms
from core.device import max_cuda_memory_allocated

# logger
from loguru import logger


def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


class Meter:
    def __init__(self):
        self._states = OrderedDict()

    def register_state(self, name, tensor):
        if name not in self._states and isinstance(tensor, torch.Tensor):
            self._states[name] = tensor

    def __getattr__(self, item):
        if "_states" in self.__dict__:
            _states = self.__dict__["_states"]
            if item in _states:
                return _states[item]

    def reset(self):
        for state in self._states.values():
            state.zero_()

    def state_dict(self):
        return dict(self._states)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self._states:
                self._states[k].copy_(v)
            else:
                raise KeyError("Unexpected key {k} in state dict when loading \
                                {self.__class__.__name__} from state dict")


class ConstantMeter(Meter):
    def __init__(self, shape):
        super(ConstantMeter, self).__init__()
        self.register_state("last", torch.zeros(shape, dtype=torch.float32))

    def update(self, value):
        self.last.copy_(value)

    @property
    def value(self):
        return self.last


class AverageMeter(ConstantMeter):
    def __init__(self, shape=(), momentum=1.):
        super(AverageMeter, self).__init__(shape)
        self.register_state("sum",      torch.zeros(
            shape,  dtype=torch.float32))
        self.register_state("count",    torch.tensor(
            0,     dtype=torch.float32))

        self.momentum = momentum

    def update(self, value):
        super(AverageMeter, self).update(value)
        self.sum.mul_(self.momentum).add_(value)
        self.count.mul_(self.momentum).add_(1.)

    @property
    def mean(self):
        if self.count.item() == 0:
            return torch.tensor(0.)
        else:
            return self.sum / self.count.clamp(min=1)


class Writer:
    """ Base class for writers """
    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class EventWriter(Writer):
    """ Event writer """

    def __init__(self, directory, print_freq=10):
        
        # summary board
        self.summray = tensorboard.SummaryWriter(directory, max_queue=5, flush_secs=10)

        # print frequency
        self.print_freq = print_freq

        # histories
        self.histories = OrderedDict()
        for mode in ["train", "eval", "test"]:
            self.histories[mode] = defaultdict(AverageMeter)

        # mode
        self.is_training = True

        # transforms
        self.inv_transform = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229,       1/0.224,       1/0.225])

    def train(self):
        self.mode = "train"
        self.is_training = True
        self.history = self.histories["train"]

    def eval(self):
        self.mode = "eval"
        self.is_training = False
        self.history = self.histories["eval"]

    def test(self):
        self.mode = "test"
        self.is_training = False
        self.history = self.histories["test"]

    def add_scalar(self, name, value, iter):
        # summary board
        self.summray.add_scalar(name, value, iter)

    def add_scalars(self, data, iter):
        print(data)
        for k, v in data.items():
            self.add_scalar(self.mode + "/" + k, v, iter)

    def add_images(self, images, iter):
        if iter % self.print_freq == 0:
            for id, im in enumerate(images):
                im = self.inv_transform(im)
                self.summray.add_images(f'tuple/{id}', im, iter)

    def add_graph(self, model, images=None):
        images = images[0].cuda()
        self.summray.add_graph(model, images)

    def put(self, name, value):
        # update history
        history = self.history[name]
        history.update(value)

    def get(self):
        return dict(self.history)

    def set(self, state_dicts):

        self.train()

        for key in ['loss', 'total_loss', 'data_time', 'batch_time']:
            self.history[key].load_state_dict(state_dicts[key])

    def write(self, global_step):
        for k, v in self.history.items():
            if isinstance(v, AverageMeter):
                self.summray.add_scalar(
                    self.mode + "/" + k, v.value.item(), global_step)

    def log(self, epoch, num_epochs, step, num_steps):
        """ Log current status """

        # epoch and steps
        msg = _current_total_formatter(epoch, num_epochs) \
                + " " + _current_total_formatter(step, num_steps)

        # metrics
        for k, v in self.history.items():
            if isinstance(v, AverageMeter):
                msg += "  {}={:.2f} ({:.2f})".format(k,
                                                   v.value.item(),
                                                   v.mean.item())
        # memory usage megabites
        if torch.cuda.is_available():
            max_mem_mb = max_cuda_memory_allocated()
            msg += f"  mem={max_mem_mb:.2f}"

        # log
        logger.info(msg)
