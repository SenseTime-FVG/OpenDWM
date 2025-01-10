# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 下午2:25
# @Author  : Hcyang
# @File    : __init__.py.py
# @Desc    : Condition utils


class AtomicCondition(object):
    def __init__(self, debug=False):
        self.debug = debug

    def check_condition_by_waypoint(self, wp):
        """
        判断当前waypoint是否满足场景触发条件
        """
        raise NotImplementedError

    def dprint(self, info):
        if self.debug:
            print(info)
