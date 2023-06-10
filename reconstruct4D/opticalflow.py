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
        self.flow_files = sorted([os.path.join(self.flow_dir, file) for file in os.listdir(self.flow_dir) if file.endswith('.flo')])
        print(f"flow_files={self.flow_files}")

    def compute(self, flow_file):
        '''
        compute optical flow from 2 consecutive images.
        currently just read flow from files.
        '''
        flow = utils.frame_utils.readFlow(flow_file)

        # debug
        flow_img = utils.flow_viz.flow_to_image(flow)
        cv2.imshow('flow', flow_img)
        


# %%