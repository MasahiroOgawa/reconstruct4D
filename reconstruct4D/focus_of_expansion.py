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

        # randomly select 2 points from flow
        for _ in range(100):
            i, j = np.random.randint(0, flow.shape[0]), np.random.randint(0, flow.shape[1])
            l1 = self.comp_flowline(i, j)
            i, j = np.random.randint(0, flow.shape[0]), np.random.randint(0, flow.shape[1])
            l2 = self.comp_flowline(i, j)
            foe = np.cross(l1, l2)
            self.draw_homogeneous_point(foe)

        # draw flow image
        cv2.imshow('flow arrow', self.flow_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()

    def comp_flowline(self, i: int, j: int):
        x = [j, i, 1]
            
        # flow
        u = self.flow[i, j, 0]
        v = self.flow[i, j, 1]
        x_prev = [j - u, i - v, 1]

        # debug. draw arrow
        cv2.arrowedLine(self.flow_img, list(map(int, x_prev[0:2])), x[0:2], (0, 0, 255), 1)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line


    def draw_homogeneous_point(self, hom_pt):
        if hom_pt[2] == 0:
            return
        pt = (int(hom_pt[0] / hom_pt[2]), int(hom_pt[1] / hom_pt[2]))
        # draw cross on the point
        cv2.line(self.flow_img, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 10)
        cv2.line(self.flow_img, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 10)
