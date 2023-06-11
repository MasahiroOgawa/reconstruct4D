import numpy as np
import cv2

class FoE():
    def __init__(self, f) -> None:
        self.f = f

    def compute(self, flow, flow_img):
        '''
        compute focus of expansion from optical flow.
        '''
        
        # compute arrow for every 10 pixels
        for i in range(0, flow.shape[0], 10):
            for j in range(0, flow.shape[1], 10):
                 x = [j, i, 1]
                 
                 # flow
                 u = flow[i, j, 0]
                 v = flow[i, j, 1]
                 x_prev = [j - u, i - v, 1]

                 # debug. draw arrow
                 cv2.arrowedLine(flow_img, list(map(int, x_prev[0:2])), x[0:2], (0, 0, 255), 1)
                 cv2.imshow('flow arrow', flow_img)

                 # no rotation correction version
                 l1 = np.cross(x, x_prev)

