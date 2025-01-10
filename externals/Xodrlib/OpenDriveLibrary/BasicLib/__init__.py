# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:47
# @Author  : Hcyang
# @File    : __init__.py.py
# @Desc    : TODO:


from typing import List
from easydict import EasyDict as Edict


class OpenDriveParser:
    def __init__(self, v):
        self.v = v

    def __getattr__(self, name):
        if name.endswith('_s'):
            items = self.v.findall(name.replace('_s', ''))
            return OpenDriveParserList([OpenDriveParser(item) for item in items])
        else:
            item = self.v.find(name)
            return OpenDriveParser(item)

    @property
    def attrib(self):
        return Edict(self.v.attrib)

    @property
    def children(self):
        return OpenDriveParserList([OpenDriveParser(item) for item in self.v.getchildren()])

    def __contains__(self, tag):
        return self.v.find(tag) is not None

    def __str__(self):
        if self.v is None:
            return f'None'
        info = f'<{self.v.tag}'
        for k, v in self.v.attrib.items():
            info += f' {k}="{v}"'
        info += '>'
        return info

    def __repr__(self):
        return self.__str__()


class OpenDriveParserList:
    def __init__(self, open_drive_parser_list: List[OpenDriveParser]):
        self.vs = open_drive_parser_list

    def find(self, **kwargs):
        result = []
        for item in self.vs:
            for k, v in kwargs.items():
                if k in item.attrib and item.attrib[k] == v:
                    result.append(item)
        return OpenDriveParserList(result)

    def first(self):
        if len(self.vs) > 0:
            return self.vs[0]
        return None

    def __str__(self):
        info = ''
        for index, item in enumerate(self.vs):
            info += f'{index}: {item}\n'
        info += f'Total: {len(self.vs)}'
        return info

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.vs)

    def __iter__(self):
        for item in self.vs:
            yield item

