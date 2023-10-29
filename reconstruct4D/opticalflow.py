import inspect
import sys
import os
import cv2
import numpy as np
import torch
from ext.unimatch import utils
from ext.unimatch.utils import frame_utils
from ext.unimatch.utils import flow_viz


class UnimatchFlow():
    '''
    compute optical flow using unimatch algorithm
    '''

    def __init__(self, FLOW_RESULT_DIR) -> None:
        self.FLOW_RESULT_DIR = FLOW_RESULT_DIR

    def compute(self, imgname):
        '''
        compute optical flow from 2 consecutive images.
        currently just read flow from files.
        args:
            imgname: image file name. e.g. 00000.jpg
        result: 
            self.flow: size = h x w x 2. 2 means flow vector (u,v).
            self.flow_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
        '''
        imgnum = imgname.split('.')[0]
        flow_file = os.path.join(self.FLOW_RESULT_DIR, f"{imgnum}_pred.flo")
        self.flow = utils.frame_utils.readFlow(flow_file)

        self.flow_img = utils.flow_viz.flow_to_image(self.flow)


class UndominantFlowAngleExtractor():
    def __init__(self, thre_angle=10 * np.pi / 180, loglevel=0) -> None:
        # constants
        # if flow length is lower than this value, the flow is ignored.
        self.thre_flowlength = 2.0

        # variables
        self.loglevel = loglevel
        self.thre_angle = thre_angle  # radian

    def compute(self, flow: np.ndarray, nonsky_static_mask: np.ndarray):
        '''
        compute undominant orientation mask from optical flow.
        args:
            flow: size = h x w x 2. 2 means flow vector (u,v).
        result:
            self.undominant_flow_prob: size = h x w.
        '''
        # compute flow angle and length
        flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        flow_length = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

        # extract median angle
        median_angle = np.median(flow_angle[nonsky_static_mask])

        # compute mask from median angle.
        self.undominant_flow_prob = np.zeros(
            (flow.shape[0], flow.shape[1]), dtype=np.float16)

        self.undominant_flow_prob[(flow_length > self.thre_flowlength) & (
            np.abs(flow_angle - median_angle) > self.thre_angle)] = 0.9 # 0.9 means outlier
        self.undominant_flow_prob[(flow_length > self.thre_flowlength) & (np.abs(flow_angle - median_angle)
                       <= self.thre_angle)] = 0.1  # 0.1 means inlier

