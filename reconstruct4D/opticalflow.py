# %%
import torch
import os
import sys
import cv2
sys.path.append('../unimatch') # for import dataloader
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
        '''
        imgnum = imgname.split('.')[0]
        flow_file = os.path.join(self.flow_dir, f"{imgnum}_pred.flo")
        flow = utils.frame_utils.readFlow(flow_file)

        self.flow_img = utils.flow_viz.flow_to_image(flow)

# %%