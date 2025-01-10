# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午4:47
# @Author  : Hcyang
# @File    : Basic.py
# @Desc    : TODO:

import os
import sys
import json
# import ipdb
import pickle
import argparse

# from tqdm import tqdm


class BasicMode(object):
    def __init__(self, args):
        self.args = args

    def run(self, *args, **kwargs):
        raise NotImplementedError
