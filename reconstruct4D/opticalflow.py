import cv2
import numpy as np
import os
import sys
import torch
sys.path.append('../ext/unimatch') # for import dataloader
import main_flow
import unimatch
import utils


class UnimatchFlow():
    '''
    compute optical flow using unimatch algorithm
    '''
    def __init__(self) -> None:
        currentfiledirectory = os.path.dirname(os.path.abspath(__file__))
        self.flow_dir = os.path.join(currentfiledirectory, '../unimatch/output/todaiura')


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
        flow_file = os.path.join(self.flow_dir, f"{imgnum}_pred.flo")
        self.flow = utils.frame_utils.readFlow(flow_file)

        self.flow_img = utils.flow_viz.flow_to_image(self.flow)



class FlowAnalyzer():
    def __init__(self, loglevel = 0) -> None:
        self.loglevel = loglevel
        self.angle_thre = 10 * np.pi / 180 # 10 degree
        pass

    def compute(self, flow: np.ndarray):
        '''
        compute undominant orientation mask from optical flow.
        args:
            flow: size = h x w x 2. 2 means flow vector (u,v).
        result:
            self.flow_mask: size = h x w. mask value: 0: unknown, 1: inlier, 2: outlier
        ''' 
        # compute flow angle
        flow_angle = np.arctan2(flow[:,:,1], flow[:,:,0])

        # extract median angle
        median_angle = np.median(flow_angle)

        # compute mask from median angle. 
        self.flow_mask = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)
        self.flow_mask[np.abs(flow_angle - median_angle) > self.angle_thre] = 2
        self.flow_mask[np.abs(flow_angle - median_angle) <= self.angle_thre] = 1

        if self.loglevel > 2:
            cv2.imshow('flow analyzer', self.flow_mask.numpy())
            cv2.waitKey(1)



def draw_flow_mask(flow_mask):
    '''
    draw flow mask image.
    args:
        flow_mask: size = h x w. mask value: 0: unknown, 1: inlier, 2: outlier
    result:
        self.result_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
    '''
    mask_img = np.zeros((flow_mask.shape[0], flow_mask.shape[1], 3), dtype=np.uint8)
    mask_img[flow_mask == 1] = (0, 255, 0)
    mask_img[flow_mask == 2] = (0, 0, 255)
    cv2.imshow("flow mask", mask_img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()
