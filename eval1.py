#!/home/simon/.pyenv/versions/rospython/bin/ipython -i
from __future__ import print_function
from __future__ import division
from typing import *
from enum import Enum

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer
from autolab_core import YamlConfig
from gqcnn import GQCNNPredictionVisualizer

import cnn_util as util
from cnn_util import DatasetSlice

def explore():
    """just some dataset exploration
    """
    modelpath = util.getEnv('gqcnn_train_stable_model', predicates=[os.path.exists], failIfNot=True)
    dataset = DatasetSlice(0)
    dataset.inspect()

def show(obj, datasetNr=0):
    content = DatasetSlice(datasetNr).getObj(obj)
    if len(content.shape) > 1:
        'images for %s ' % obj
        util.iiBrowse( content )
    if len(content.shape) == 1:
        'plot for %s ' % obj
        plt.plot(content)

def visualization():
    visualization_config = YamlConfig('/home/simon/sandbox/graspitmod/catkin_ws/src/gqcnn/custom/visualization.yaml')

    visualizer = GQCNNPredictionVisualizer(visualization_config)
    visualizer.visualize()

def predict():
    modelpath = util.getEnv('gqcnn_train_stable_model', predicates=[os.path.exists], failIfNot=True)
    """
    The images should be specified as an `N`x32x32x1 array and the poses should be specified as an `N`x1 array of depths, where `N` is the number of datapoints to predict.
    For an example, load a batch of images from `depth_ims_tf_table_00000.npz` and a batch of corresponding poses from column 2 of `hand_poses_00000.npz` from the Adv-Synth dataset.
    pred_p_success = output[:,1]
    """
    gqcnn = GQCNN.load(modelpath)
    imgs = DatasetSlice(0).getObj(DatasetSlice.obj_depth_ims_tf_table)
    img = imgs[0]
    poses = DatasetSlice(0).getObj(DatasetSlice.obj_hand_poses)
    pose = poses[0]
    # prediction = gqcnn.predict(np.array([img]), np.array([pose[1]]))

    print ( poses[:,1].shape )
    # sys.exit()
    prediction = gqcnn.predict(imgs, poses[:, 1])
    print('prediction is %s' % prediction)

# functions: (ex)
# explore()
# show(DatasetSlice.obj_depth_ims_tf_table)
# visualize()
# predict()


# predictions =============
# sys.exit(0)

