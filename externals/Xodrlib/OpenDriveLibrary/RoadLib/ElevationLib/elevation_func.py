
from OpenDriveLibrary.BasicLib import OpenDriveParser, OpenDriveParserList
import numpy as np
from typing import List

class ElevationParser(object):
    # Desc: 道路单个高程
    def __init__(self, elevation: OpenDriveParser, road):
        self.elevation = elevation
        self.road = road
        self.s0 = float(elevation.attrib.s)
        self.a = float(elevation.attrib.a)
        self.b = float(elevation.attrib.b)
        self.c = float(elevation.attrib.c)
        self.d = float(elevation.attrib.d)

    def s2z(self, s):
        ds = s - self.s0
        z = self.a + self.b * ds + self.c * np.power(ds, 2) + self.d * np.power(ds, 3)
        return z

    def __str__(self):
        return str(self.elevation)

    def __repr__(self):
        return self.__str__()


class ElevationListParser(object):
    # Desc: 道路几何中心线集合
    def __init__(self, elevations: List[ElevationParser], road):
        self.elevations = elevations
        self.road = road

    def __iter__(self):
        for elevation in self.elevations:
            yield elevation

    def __str__(self):
        info = ''
        for index, elevation in enumerate(self.elevations):
            info += f'{index}: {elevation}\n'
        return info

    def __repr__(self):
        return self.__str__()
