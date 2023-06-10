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
        flow_dir = '../unimatch/output/todaiura'
        self.flow_files = sorted([os.path.join(flow_dir, file) for file in os.listdir(flow_dir) if file.endswith('.flo')])
        print(f"flow_files={self.flow_files}")

    def compute(self, img1, img2):
        '''
        compute optical flow from 2 consecutive images.
        currently just read flow from files.
        '''
        for flow_file in self.flow_files:
            flow = utils.frame_utils.readFlow(flow_file)

            # debug
            img = utils.flow_viz.flow_to_image(flow)
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        


# %%