import torch
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt
from .histogram import *
from .dist_helper import compute_mmd, gaussian, compute_mmd_sigma, gaussian_dist
import glob
import random 
from .hist_utils import *

def jsd_2d(p, q):
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p.flatten(), q.flatten())

def calculate_jsd(sample_folder):
    model_samples = load_range_images(sample_folder)
    kitti_samples = load_kitti(len(model_samples), 0)

    model_histograms = array_to_histograms(model_samples)
    kitti_histograms = array_to_histograms(kitti_samples)

    model_p = np.stack(model_histograms, axis=0)
    model_p = np.sum(model_p, axis=0)
    model_p = model_p / np.sum(model_p)

    kitti_p = np.stack(kitti_histograms, axis=0)
    kitti_p = np.sum(kitti_p, axis=0)
    kitti_p = kitti_p / np.sum(kitti_p)

    jsd_score = jsd_2d(kitti_p, model_p)

    return jsd_score