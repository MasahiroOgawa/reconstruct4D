import numpy as np
import cv2
from enum import Enum

CameraState = Enum('CameraState', ['STOPPING', 'ROTATING', 'ONLY_TRANSLATING'])


class FoE():
    def __init__(self, loglevel=0) -> None:
        # constants
        # 0: no log, 1: print log, 2: display image, 3: debug with detailed image
        self.loglevel = loglevel
        # if flow length is lower than this value, the flow is ignored.
        self.thre_flowlength = 2.0
        # if angle between flow and foe is lower than this value, the flow is inlier.[radian]
        self.thre_inlier_angle = 10 * np.pi / 180
        # if inlier rate is higher than this value, the foe is accepted.
        self.thre_inlier_rate = 0.9
        # if flow existing pixel rate is lower than this value, the camera is considered as stopping.
        # the flow existing rate will be computed only inside static mask.
        self.thre_flow_existing_rate = 0.1
        self.num_ransac = 10
        self.state = CameraState.ROTATING  # most unkown movement.
        self.flowarrow_step = 20  # every this pixel, draw flow arrow.

        # variables
        self.flow_existing_rate_in_static = 0.0
        self.inlier_rate = 0.0
        self.foe = None
        self.result_img = None
        self.inlier_mask = None  # 0: unknown, 1: inlier, 2: outlier
        self.maxinlier_mask = None
        self.debug_img = None

    def compute(self, flow, sky_mask, static_mask):
        '''..ext.
        compute focus of expansion from optical flow.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
            sky_mask: mask of sky. shape = (height, width), dtype = bool.
            static_mask: mask of static object like grounds. shape = (height, width), dtype = bool
        '''
        self.flow = flow
        self.sky_mask = sky_mask
        self.static_mask = static_mask

        self.prepare_variables()

        self.comp_flow_existing_rate()

        if self.flow_existing_rate_in_static < self.thre_flow_existing_rate:
            self.state = CameraState.STOPPING
            # at this moment, all flow existing pixel is set as outlier.
            self.maxinlier_mask = self.inlier_mask.copy()
        else:
            self.comp_foe_by_ransac()

    def draw(self, bg_img=None):
        self.bg_img = bg_img

        self.prepare_canvas()
        self.draw_state()

    def prepare_variables(self):
        self.state = CameraState.ROTATING
        self.inlier_mask = np.zeros(
            (self.flow.shape[0], self.flow.shape[1]), dtype=np.uint8)
        self.maxinlier_mask = np.zeros(
            (self.flow.shape[0], self.flow.shape[1]), dtype=np.uint8)

    def comp_flow_existing_rate(self):
        num_flow_existing_pix_in_static = 0

        # set non sky mask as outlier, that is a moving object.
        self.inlier_mask[self.sky_mask == False] = 2

        # check pixels inside static mask
        staticpix_indices = np.where(self.static_mask == True)
        for i in range(len(staticpix_indices[0])):
            row = staticpix_indices[0][i]
            col = staticpix_indices[1][i]

            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            # skip if flow is too small
            if (u**2 + v**2) < self.thre_flowlength**2:
                self.inlier_mask[row, col] = 0  # unknown and stop.
            else:
                # inlier_mask is already set as outlier;2.
                num_flow_existing_pix_in_static += 1

        self.flow_existing_rate_in_static = num_flow_existing_pix_in_static / \
            len(staticpix_indices[0])

        if self.loglevel > 0:
            print(
                f"[INFO] flow existing pixel rate: {self.flow_existing_rate_in_static * 100:.2f} %")

        if self.loglevel > 2:
            inlier_mask_img = np.zeros(
                (self.inlier_mask.shape[0], self.inlier_mask.shape[1], 3), dtype=np.uint8)
            inlier_mask_img[self.inlier_mask == 1] = [0, 255, 0]
            inlier_mask_img[self.inlier_mask == 2] = [0, 0, 255]
            cv2.imshow('non sky & stopping static mask', inlier_mask_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()

    def comp_foe_by_ransac(self):
        '''
        compute FoE by RANSAC
        '''
        max_inlier_rate = 0.0

        for _ in range(self.num_ransac):
            foe_candi = self.comp_foe_candidate()
            if foe_candi is None:
                continue

            self.comp_inlier_rate(foe_candi)

            if self.inlier_rate > max_inlier_rate:
                # update by the current best
                max_inlier_rate = self.inlier_rate
                self.maxinlier_mask = self.inlier_mask.copy()
                # currently we don't recompute FoE using all inliers, because our final objective is getting outlier mask.
                self.foe = foe_candi

                # stop if inlier rate is high enough
                if self.inlier_rate > self.thre_inlier_rate:
                    self.state = CameraState.ONLY_TRANSLATING
                    break

    def comp_foe_candidate(self) -> np.ndarray:
        '''
        compute FoE from 2 flow lines only inside static mask.
        FoE is in 3D homogeneous coordinate.
        '''
        # compute FoE from 2 flow lines only inside static mask.
        row, col = self.random_point_in_static_mask()
        l1 = self.comp_flowline(row, col)
        if l1 is None:
            return None
        row, col = self.random_point_in_static_mask()
        l2 = self.comp_flowline(row, col)
        if l2 is None:
            return None
        foe = np.cross(l1, l2)

        # draw debug image
        if self.loglevel > 2:
            self.debug_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8)
            self.draw_line(l1, self.debug_img)
            self.draw_line(l2, self.debug_img)
            self.draw_homogeneous_point(foe, self.debug_img)
            cv2.imshow('Debug', self.debug_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()

        return foe

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
        if (u**2 + v**2) < self.thre_flowlength**2:
            return None

        x_prev = [col - u, row - v, 1]

        if self.loglevel > 2:
            cv2.arrowedLine(self.debug_img, tuple(
                map(int, x_prev[0:2])), tuple(x[0:2]), (0, 0, 255), 3)

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line

    def random_point_in_static_mask(self):
        # Find the indices of all pixels in the static mask that have a value of 1
        indices = np.where(self.static_mask == 1)

        # Randomly select one of the indices
        index = np.random.choice(len(indices[0]))

        # Get the row and column values corresponding to the selected index
        row = indices[0][index]
        col = indices[1][index]

        return row, col

    def comp_inlier_rate(self, foe) -> float:
        '''
        compute inlier rate except sky mask from FoE 
        args: 
            foe: FoE in 3D homogeneous coordinate
        '''
        num_inlier = 0
        num_flow_existingpix = 0
        thre_cos = np.cos(self.thre_inlier_angle)

        # treat candidate FoE is infinite case
        if foe[2] == 0:
            foe[2] = 1e-10
        else:
            foe_u = foe[0] / foe[2]
            foe_v = foe[1] / foe[2]

        # check pixels inside "non-static" & "moving static" object area.
        for row, col in zip(*np.nonzero(self.inlier_mask)):
            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            # skip if flow is too small
            if (u**2 + v**2) < self.thre_flowlength**2:
                # this means nonstatic object which moves with camera.
                # camera is not stopping, so the object must be moving.
                self.inlier_mask[row, col] = 2
            else:
                num_flow_existingpix += 1

                if self.loglevel > 2:
                    foe_flow_img = self.debug_img.copy()
                    cv2.arrowedLine(foe_flow_img, (int(foe_u), int(foe_v)),
                                    (col, row), (0, 255, 0), 3)
                    cv2.arrowedLine(foe_flow_img, (col, row),
                                    (int(col+u), int(row+v)), (0, 0, 255), 3)
                    cv2.imshow('Debug', foe_flow_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        exit()

                # check the angle between flow and FoE to each pixel is lower than threshold.
                cos_foe_flow = np.dot((col-foe_u, row-foe_v), (u, v)) / \
                    np.sqrt((col-foe_u)**2 + (row-foe_v)**2) / \
                    np.sqrt(u**2 + v**2)
                if cos_foe_flow > thre_cos:
                    num_inlier += 1
                    self.inlier_mask[row, col] = 1  # inlier
                else:
                    self.inlier_mask[row, col] = 2  # outlier

        if self.flow_existing_rate_in_static == 0:
            self.inlier_rate = 0
        else:
            self.inlier_rate = num_inlier / num_flow_existingpix

        if self.loglevel > 0:
            print(
                f"[INFO] FoE candidate: {foe}, inlier rate: {self.inlier_rate * 100:.2f} %")

    def prepare_canvas(self):
        if self.bg_img is None:
            self.result_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8)
        else:
            self.result_img = self.bg_img.copy()
            if self.loglevel > 1:
                self.draw_flowarrow(self.flow, self.result_img)
            if self.loglevel > 2:
                self.debug_img = self.bg_img.copy()

    def draw_flowarrow(self, flow, img):
        '''
        draw flow as arrow
        '''
        for row in range(0, flow.shape[0], self.flowarrow_step):
            for col in range(0, flow.shape[1], self.flowarrow_step):
                u = flow[row, col, 0]
                v = flow[row, col, 1]
                cv2.arrowedLine(img, pt1=(int(col-u), int(row-v)),
                                pt2=(col, row), color=(0, 0, 255), thickness=3)

    def draw_homogeneous_point(self, hom_pt, out_img):
        if hom_pt[2] == 0:
            return
        pt = (int(hom_pt[0] / hom_pt[2]), int(hom_pt[1] / hom_pt[2]))
        # draw cross at the point
        cv2.line(out_img, (pt[0] - 10, pt[1]),
                 (pt[0] + 10, pt[1]), (0, 0, 255), 10)
        cv2.line(out_img, (pt[0], pt[1] - 10),
                 (pt[0], pt[1] + 10), (0, 0, 255), 10)

    def draw_line(self, line, img):
        if line[0] == 0 and line[1] == 0:
            return
        pt1 = (0, int(-line[2] / line[1]))
        pt2 = (img.shape[1],
               int(-(line[2] + line[0] * img.shape[1]) / line[1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)

    def draw_state(self):
        if self.state == CameraState.STOPPING:
            cv2.putText(self.result_img, "Camera is stopping",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        elif self.state == CameraState.ONLY_TRANSLATING:
            self.draw_homogeneous_point(self.foe, self.result_img)
            cv2.putText(self.result_img, "Camera is only translating",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        elif self.state == CameraState.ROTATING:
            cv2.putText(self.result_img, "Camera is rotating",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
