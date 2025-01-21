# Copyright 2024 Vchitect/Latte

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.# Modified from Latte

# - This file is adapted from https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py


import numbers
import random

import numpy as np
import torch


class VideoMetaCollate(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, text_duration):
        self.text_duration = text_duration
        self.debug_print = True

    def __call__(self, results):
        if "text_duration" in results:
            return results

        assert "fps" in results
        if len(results["images"]) == 1:         # not a video
            results["fps"] = 1e6
            if self.debug_print:
                print("Set FPS to 1e6 ...")
                self.debug_print = False
        if self.text_duration == "dynamic":
            results["text_duration"] = len(results["images"])*10 / results["fps"]
        else:
            assert isinstance(self.text_duration, int)
            results["text_duration"] = self.text_duration
        return results


class TextVideo2Image(object):
    def __init__(self, candidate="video_description", target="image_description",):
        self.candidate = candidate
        self.target = target

    def __call__(self, results):
        if self.candidate in results:
            assert self.target not in results
            # t, ... -> 1, ...
            results[self.target] = results[k][0:1]
            results.pop(self.candidate)

        return results


class TextImage2Video(object):
    def __init__(self, candidate="image_description", target="video_description"):
        self.candidate = candidate
        self.target = target

    def __call__(self, results):
        if self.candidate in results:
            # 1, ... -> t, ...
            assert len(results[self.candidate]) == 1
            results[self.target] = sum(
                [results[self.candidate] for _ in range(len(results["images"]))], [])
            results.pop(self.candidate)

        return results