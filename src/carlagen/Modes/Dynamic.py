# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午4:44
# @Author  : Hcyang
# @File    : Dynamic.py
# @Desc    : TODO:

import os
import sys
import json
# import ipdb
import pickle
import argparse

# from tqdm import tqdm
from .Basic import BasicMode


class DynamicMode(BasicMode):
    def __init__(self, args):
        super(DynamicMode, self).__init__(args)

    def run(self, *args, **kwargs):
        pass
