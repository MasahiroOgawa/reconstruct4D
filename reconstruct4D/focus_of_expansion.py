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
        self.foe_img = flow_img 
        self.debug_img = flow_img

        # randomly select 2 points from flow
        for _ in range(100):
            i, j = np.random.randint(0, flow.shape[0]), np.random.randint(0, flow.shape[1])
            l1 = self.comp_flowline(i, j)
            for _ in range(10):
                self.debug_img = self.foe_img.copy()

                i, j = np.random.randint(0, flow.shape[0]), np.random.randint(0, flow.shape[1])
                l2 = self.comp_flowline(i, j)
                foe = np.cross(l1, l2)

                # debug. draw line
                self.draw_line(l1)
                self.draw_line(l2)
                self.draw_homogeneous_point(foe, self.debug_img)
                cv2.imshow('FoE', self.debug_img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    exit()

        # draw flow image
        cv2.imshow('foe', self.foe_img)
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
        cv2.arrowedLine(self.debug_img, list(map(int, x_prev[0:2])), x[0:2], (0, 0, 255), 3)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line


    def draw_homogeneous_point(self, hom_pt, out_img):
        if hom_pt[2] == 0:
            return
        pt = (int(hom_pt[0] / hom_pt[2]), int(hom_pt[1] / hom_pt[2]))
        # draw cross at the point
        cv2.line(out_img, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 10)
        cv2.line(out_img, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 10)

    def draw_line(self, line):
        if line[0] == 0 and line[1] == 0:
            return
        pt1 = (0, int(-line[2] / line[1]))
        pt2 = (self.foe_img.shape[1], int(-(line[2] + line[0] * self.foe_img.shape[1]) / line[1]))
        cv2.line(self.debug_img, pt1, pt2, (0, 255, 0), 1)