# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 下午4:56
# @Author  : Hcyang
# @File    : __init__.py.py
# @Desc    : xxx


from random import choice


# Desc: Apollo所有决策空间
KEEP_L = 'FOLLOW'
LEFT_C = 'LEFT_CHANGE'
RIGHT_C = 'RIGHT_CHANGE'
LEFT_B = 'LEFT_BORROW'
RIGHT_B = 'RIGHT_BORROW'
ACC = 'ACCELERATE'
DEC = 'DECELERATE'
KEEP_S = 'KEEP'
STOP = 'STOP'
UNKNOWN = 'UNKNOWN'

class BasicDesc(object):
    def __init__(self, world, ego):
        self.world = world
        self.ego = ego
        self.atomic_desc = {}
        self.candidate_reason = {'path': {}, 'speed': {}}

    def get_env_desc(self, short=False, long=False, **kwargs):
        """
        Desc: 获取描述
        Special: [短描述，长描述，反面描述]
        """
        self._reset_candidate_reason()
        self.atomic_desc = self._gen_atomic_desc(**kwargs)
        env_desc = self._gen_env_desc(short=short, long=long)
        return env_desc

    def get_path_reason(self, path_decision, random_one=False, short=True, long=False):
        """
        Desc: 获取路径决策的原因，默认组合所有原因
        Special: 选择短词
        :param path_decision: 路径决策
        :param random_one: 是否随机选一个原因
        :param short: 是否短词
        :param long: 是否长词
        """
        if path_decision not in self.candidate_reason['path']:
            return ''
        reasons = self.candidate_reason['path'][path_decision]
        if random_one:
            reasons = [choice(reasons)]
        if short:
            reasons = [self.atomic_desc[k][0] for k in reasons]
        elif long:
            reasons = [self.atomic_desc[k][1] for k in reasons]
        else:
            raise ValueError('short or long should be True')
        return [item for item in reasons if item.strip()]

    def get_speed_reason(self, speed_decision, random_one=False, short=True, long=False):
        """
        Desc: 获取速度决策的原因，默认组合所有原因
        Special: 选择短词
        :param speed_decision: 速度决策
        :param random_one: 是否随机选一个原因
        """
        if speed_decision not in self.candidate_reason['speed']:
            return ''
        reasons = self.candidate_reason['speed'][speed_decision]
        if random_one:
            reasons = [choice(reasons)]
        if short:
            reasons = [self.atomic_desc[k][0] for k in reasons]
        elif long:
            reasons = [self.atomic_desc[k][1] for k in reasons]
        else:
            raise ValueError('short or long should be True')
        return [item for item in reasons if item.strip()]

    def _gen_atomic_desc(self, **kwargs):
        raise NotImplementedError

    def _gen_env_desc(self, **kwargs):
        raise NotImplementedError

    def _reset_candidate_reason(self):
        self.candidate_reason = {'path': {}, 'speed': {}}

    def add_path_reason(self, path_decision, reason):
        self.candidate_reason['path'].setdefault(path_decision, []).append(reason)

    def add_speed_reason(self, speed_decision, reason):
        self.candidate_reason['speed'].setdefault(speed_decision, []).append(reason)
