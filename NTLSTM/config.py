import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict


class edict(OrderedDict):
    """Using OrderedDict for the `easydict` package
    See Also https://pypi.python.org/pypi/easydict/
    """

    def __init__(self, d=None, **kwargs):
        super(edict, self).__init__()
        if d is None:
            d = OrderedDict()
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        # special handling of self.__root and self.__map
        if name.startswith('_') and (name.endswith('__root') or name.endswith('__map')):
            super(edict, self).__setattr__(name, value)
        else:
            if isinstance(value, (list, tuple)):
                value = [self.__class__(x)
                         if isinstance(x, dict) else x for x in value]
            else:
                value = self.__class__(value) if isinstance(value, dict) else value
            super(edict, self).__setattr__(name, value)
            super(edict, self).__setitem__(name, value)

    __setitem__ = __setattr__

class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 1
__C.GLOBAL.MODEL_CONVLSTM = 'convLSTM'

__C.HKO = edict()

__C.HKO.EVALUATION = edict()
# __C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
__C.HKO.EVALUATION.THRESHOLDS = np.array([20, 30, 40, 50])
__C.HKO.EVALUATION.MIDDLE_VALUE = np.array([10, 25, 35, 45, 65])
__C.HKO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)
# __C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 3, 10, 20)

__C.HKO.EVALUATION.VALID_DATA_USE_UP = True
# __C.HKO.EVALUATION.VALID_TIME = 100
__C.HKO.EVALUATION.VALID_TIME = 20

__C.HKO.BENCHMARK = edict()

__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 120  # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 24  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5  # The stride


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 256
__C.HKO.ITERATOR.HEIGHT = 256


__C.MODEL = edict()
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
