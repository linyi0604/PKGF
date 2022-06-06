import sys
from torch.nn import init
import torch.nn as nn
from albumentations import Compose, RandomRotate90, Flip, Transpose, \
    OneOf, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, \
        Blur, ShiftScaleRotate, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, \
            CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, \
                HueSaturationValue, Resize
                

# evaluate meters
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


# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        if m.bias is not None:
            init.constant(m.bias, 0)



def data_augmentadion(p=0.5, train=True):
    if train:
        T =  Compose([
            Resize(height=224, width=224, p=1),
            RandomRotate90(p=p),
            Flip(p=p),
            Transpose(p=p),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2*p),
            OneOf([
                MotionBlur(p=0.2*p),
                MedianBlur(blur_limit=3, p=0.1*p),
                Blur(blur_limit=3, p=0.1*p),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2*p),
            OneOf([
                OpticalDistortion(p=0.3*p),
                GridDistortion(p=0.1*p),
                IAAPiecewiseAffine(p=0.3*p),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3*p),
            HueSaturationValue(p=0.3*p),
        ], p=1) 
    else:
        T = Compose([
                Resize(height=224, width=224, p=1),
        ])
    return T



