
import torch
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt
from .histogram import *
from .dist_helper import compute_mmd, gaussian, compute_mmd_sigma, gaussian_dist
import glob
import random 
from .hist_utils import *

def calculate_mmd(sample_folder):
    model_samples = load_range_images(sample_folder)
    kitti_samples = load_kitti(len(model_samples), 0)

    model_histograms = array_to_histograms(model_samples)
    kitti_histograms = array_to_histograms(kitti_samples)

    kitti_model_distance = compute_mmd(kitti_histograms, model_histograms, gaussian, is_hist=True)

    return kitti_model_distance