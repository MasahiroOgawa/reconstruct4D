import numpy as np
import cv2

class FoE():
    def __init__(self, f) -> None:
        self.f = f

    def compute(self, flow, flow_img):
        '''
        compute focus of expansion from optical flow.
        '''
        self.flow = flow
        self.flow_img = flow_img 

        # compute arrow for every 10 pixels
        for i in range(0, flow.shape[0], 10):
            for j in range(0, flow.shape[1], 10):
                self.comp_flowline(i, j)



    def comp_flowline(self, i: int, j: int):
        x = [j, i, 1]
            
        # flow
        u = self.flow[i, j, 0]
        v = self.flow[i, j, 1]
        x_prev = [j - u, i - v, 1]

        # debug. draw arrow
        cv2.arrowedLine(self.flow_img, list(map(int, x_prev[0:2])), x[0:2], (0, 255, 0), 1)
        cv2.imshow('flow arrow', self.flow_img)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line
