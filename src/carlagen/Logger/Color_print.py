# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 下午4:21
# @Author  : Hcyang
# @File    : Color_print.py
# @Desc    : TODO:


class ColorPrint(object):
    def __init__(self):
        self.RED = '\033[91m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.PURPLE = '\033[95m'
        self.CYAN = '\033[96m'
        self.END = '\033[0m'

        self.ERROR = self.RED
        self.SUCCESS = self.GREEN
        self.WARNING = self.YELLOW
        self.INFO = self.BLUE
        self.SPECIAL = self.PURPLE
        self.MESSAGE = self.CYAN


def error(msg):
    print(f'{ColorPrint().ERROR}{msg}{ColorPrint().END}')


def success(msg):
    print(f'{ColorPrint().SUCCESS}{msg}{ColorPrint().END}')


def warning(msg):
    print(f'{ColorPrint().WARNING}{msg}{ColorPrint().END}')


def info(msg):
    print(f'{ColorPrint().INFO}{msg}{ColorPrint().END}')


def special(msg):
    print(f'{ColorPrint().SPECIAL}{msg}{ColorPrint().END}')


def message(msg):
    print(f'{ColorPrint().MESSAGE}{msg}{ColorPrint().END}')
