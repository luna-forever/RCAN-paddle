import os

import common

import numpy as np
# import imageio as misc
# misc.imread or misc.imsave are not available in the newer version of scipy. Instead, try to use imageio
import imageio

import paddle
from paddle.io import Dataset

class Demo(Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'Demo'
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = False

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        #lr = misc.imread(self.filelist[idx])
        lr = imageio.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return common.np2Tensor([lr], self.args.rgb_range)[0], -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

