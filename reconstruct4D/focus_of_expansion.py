import numpy as np
import cv2

loglevel = 3 # 0: no log, 1: print log, 2: debug, 3: debug with detailed image

class FoE():
    def __init__(self, f) -> None:
        self.f = f # focal length
        self.flow_thre = 3.0 # if u and v flow is lower than this value, the flow is ignored.


    def compute(self, flow, flow_img = None):
        '''
        compute focus of expansion from optical flow.
        '''
        self.flow = flow

        if loglevel>1:
            self.foe_img = flow_img.copy()
            self.debug_img = flow_img.copy()
            self.draw_flowarrow(flow, self.foe_img)

        # randomly select 2 points from flow
        for _ in range(100):
            foe = self.comp_foe_candidate()

        if loglevel>1:
            cv2.imshow('FoE', self.foe_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()


    def comp_flowline(self, row: int, col: int):
        x = [col, row, 1]
            
        # flow
        u = self.flow[row, col, 0]
        v = self.flow[row, col, 1]

        # if flow is too small, return zero line
        if abs(u) < self.flow_thre and abs(v) < self.flow_thre:
            return [0, 0, 0]

        x_prev = [col - u, row - v, 1]

        if loglevel>2:
            cv2.arrowedLine(self.debug_img, list(map(int, x_prev[0:2])), x[0:2], (0, 0, 255), 3)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line
    

    def comp_foe_candidate(self):
        # compute FoE from 2 flow lines
        row, col = np.random.randint(0, self.flow.shape[0]), np.random.randint(0, self.flow.shape[1])
        l1 = self.comp_flowline(row, col)
        if np.array(l1).all() == 0                               :
            return
        row, col = np.random.randint(0, self.flow.shape[0]), np.random.randint(0, self.flow.shape[1])
        l2 = self.comp_flowline(row, col)
        if np.array(l2).all() == 0:
            return
        foe = np.cross(l1, l2)

        # draw debug image
        if loglevel>2:
            self.debug_img = self.foe_img.copy()
            self.draw_line(l1, self.debug_img)
            self.draw_line(l2, self.debug_img)
            self.draw_homogeneous_point(foe, self.debug_img)
            self.draw_homogeneous_point(foe, self.foe_img)
            cv2.imshow('Debug', self.debug_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()

        return foe


    def draw_homogeneous_point(self, hom_pt, out_img):
        if hom_pt[2] == 0:
            return
        pt = (int(hom_pt[0] / hom_pt[2]), int(hom_pt[1] / hom_pt[2]))
        # draw cross at the point
        cv2.line(out_img, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 10)
        cv2.line(out_img, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 10)


    def draw_line(self, line, img):
        if line[0] == 0 and line[1] == 0:
            return
        pt1 = (0, int(-line[2] / line[1]))
        pt2 = (img.shape[1], int(-(line[2] + line[0] * img.shape[1]) / line[1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)


    def draw_flowarrow(self, flow, img):
        '''
        draw flow as arrow
        '''
        for row in range(0, flow.shape[0], 10):
            for col in range(0, flow.shape[1], 10):
                u = flow[row, col, 0]
                v = flow[row, col, 1]
                cv2.arrowedLine(img, (int(col-u), int(row-v)), (col, row), (0, 0, 255), 1)