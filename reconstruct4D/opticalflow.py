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

    def __init__(self, flow_result_dir) -> None:
        self.flow_result_dir = flow_result_dir

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
        flow_file = os.path.join(self.flow_result_dir, f"{imgnum}_pred.flo")
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
        pass

    def compute(self, flow: np.ndarray, nonsky_static_mask: np.ndarray):
        '''
        compute undominant orientation mask from optical flow.
        args:
            flow: size = h x w x 2. 2 means flow vector (u,v).
        result:
            self.flow_mask: size = h x w. mask value: 0: unknown, 1: inlier, 2: outlier
        '''
        # compute flow angle and length
        flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        flow_length = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

        # TODO: compute rotation center and extract different moving area.

        # extract median angle
        median_angle = np.median(flow_angle[nonsky_static_mask])

        # compute mask from median angle.
        self.flow_mask = np.zeros(
            (flow.shape[0], flow.shape[1]), dtype=np.uint8)

        self.flow_mask[(flow_length > self.thre_flowlength) & (
            np.abs(flow_angle - median_angle) > self.thre_angle)] = 2
        self.flow_mask[(flow_length > self.thre_flowlength) & (np.abs(flow_angle - median_angle)
                       <= self.thre_angle)] = 1


def flow_mask_img(flow_mask):
    '''
    draw flow mask image.
    args:
        flow_mask: size = h x w. mask value: 0: unknown, 1: inlier, 2: outlier
    result:
        self.result_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
    '''
    mask_img = np.zeros(
        (flow_mask.shape[0], flow_mask.shape[1], 3), dtype=np.uint8)
    mask_img[flow_mask < 0.5] = (0, 255, 0)
    mask_img[flow_mask > 0.5] = (0, 0, 255)
    return mask_img
