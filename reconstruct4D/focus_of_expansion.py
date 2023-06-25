import numpy as np
import cv2

loglevel = 2 # 0: no log, 1: print log, 2: display image, 3: debug with detailed image

class FoE():
    def __init__(self, f) -> None:
        # fixed parameters
        self.f = f # focal length
        self.flow_thre = 3.0 # if flow length is lower than this value, the flow is ignored.
        self.inlier_angle_thre = 10 * np.pi / 180 # if angle between flow and foe is lower than this value, the flow is inlier.[radian]
        self.inlier_rate_thre = 0.9 # if inlier rate is higher than this value, the foe is accepted.
       
        # variables
        self.inlier_rate = 0.0
        self.foe = None
        self.result_img = None


    def compute(self, flow, flow_img = None):
        '''
        compute focus of expansion from optical flow.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
        '''
        self.flow = flow

        if loglevel>1:
            self.result_img = flow_img.copy()
            self.debug_img = flow_img.copy()
            self.draw_flowarrow(flow, self.result_img)

        # randomly select 2 points from flow to compute FoE.
        for _ in range(100):
            foe_candi = self.comp_foe_candidate()
            if foe_candi is None:
                continue
            self.comp_inlier_rate(foe_candi)
            print(f"FoE candidate: {foe_candi} , inlier_rate: {self.inlier_rate * 100:.2f} %")
            if self.inlier_rate > self.inlier_rate_thre:
                self.foe = foe_candi
                self.draw_homogeneous_point(self.foe, self.result_img)
                break

        if loglevel>1:
            if self.inlier_rate < self.inlier_rate_thre:
                cv2.putText(self.result_img, "Camera is rotating", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('FoE', self.result_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()


    def comp_flowline(self, row: int, col: int) -> np.ndarray:
        '''
        compute flow line from flow at (row, col)
        args:
            row: row index of flow
            col: column index of flow
        return:
            line: flow line in 3D homogeneous coordinate. if flow is too small, return None.
        '''
        x = [col, row, 1]
            
        # flow
        u = self.flow[row, col, 0]
        v = self.flow[row, col, 1]

        # if flow is too small, return zero line
        if (u**2 + v**2) < self.flow_thre**2:
            return None

        x_prev = [col - u, row - v, 1]

        if loglevel>2:
            cv2.arrowedLine(self.debug_img, list(map(int, x_prev[0:2])), x[0:2], (0, 0, 255), 3)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line
    

    def comp_foe_candidate(self) -> np.ndarray: 
        '''
        compute FoE from 2 flow lines
        FoE is in 3D homogeneous coordinate.
        '''
        # compute FoE from 2 flow lines
        row, col = np.random.randint(0, self.flow.shape[0]), np.random.randint(0, self.flow.shape[1])
        l1 = self.comp_flowline(row, col)
        if l1 is None :
            return None
        row, col = np.random.randint(0, self.flow.shape[0]), np.random.randint(0, self.flow.shape[1])
        l2 = self.comp_flowline(row, col)
        if l2 is None :    
            return None
        foe = np.cross(l1, l2)

        # draw debug image
        if loglevel>2:
            self.debug_img = self.result_img.copy()
            self.draw_line(l1, self.debug_img)
            self.draw_line(l2, self.debug_img)
            self.draw_homogeneous_point(foe, self.debug_img)
            cv2.imshow('Debug', self.debug_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()

        return foe

    def comp_inlier_rate(self, foe) -> float:
        '''
        compute inlier rate from FoE
        args: 
            foe: FoE in 3D homogeneous coordinate
        '''
        num_inlier = 0
        num_valid_pixel = 0
        # check all pixels
        for row in range(self.flow.shape[0]):
            for col in range(self.flow.shape[1]):
                u = self.flow[row, col, 0]
                v = self.flow[row, col, 1]

                # skip if flow is too small
                if (u**2 + v**2) < self.flow_thre**2:
                    continue
                num_valid_pixel += 1

                # compare angle between flow and FoE to each pixel
                estimated_angle = np.arctan2(foe[0] - col, foe[1] - row)
                flow_angle = np.arctan2(u, v)
                if np.abs(estimated_angle - flow_angle) < self.inlier_angle_thre:
                    num_inlier += 1

        if num_valid_pixel == 0:
            self.inlier_rate = 0
        else:
            self.inlier_rate = num_inlier / num_valid_pixel

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