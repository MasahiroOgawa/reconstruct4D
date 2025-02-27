import numpy as np
import cv2
from enum import Enum
from scipy import sparse

CameraState = Enum("CameraState", ["STOPPING", "ROTATING", "ONLY_TRANSLATING"])


class FoE:
    def __init__(
        self,
        thre_flowlength=4.0,
        thre_inlier_angle=10 * np.pi / 180,
        thre_inlier_rate=0.9,
        thre_flow_existing_rate=0.1,
        num_ransac=10,
        same_flowangle_min_moving_prob=0.1,
        same_flowlength_min_moving_prob=0.4,
        flowarrow_step=20,
        ransac_all_inlier_estimation=False,
        search_step=1,
        log_level=0,
    ) -> None:
        # constants
        self.LOG_LEVEL = log_level
        self.THRE_FLOWLENGTH = thre_flowlength
        self.THRE_INLIER_ANGLE = thre_inlier_angle
        self.THRE_COS_INLIER = np.cos(self.THRE_INLIER_ANGLE)
        self.THRE_INLIER_RATE = thre_inlier_rate
        self.THRE_FLOW_EXISTING_RATE = thre_flow_existing_rate
        self.NUM_RANSAC = num_ransac
        self.FLOWARROW_STEP = flowarrow_step
        self.SAME_FLOWANGLE_MIN_MOVING_PROB = same_flowangle_min_moving_prob
        self.SAME_FLOWLENGTH_MIN_MOVING_PROB = same_flowlength_min_moving_prob
        self.RANSAC_ALL_INLIER_ESTIMATION = ransac_all_inlier_estimation
        self.SEARCH_STEP = search_step

        # variables
        self.state = CameraState.ROTATING  # most unkown movement.
        self.flow_existing_rate_in_static = 0.0
        self.mean_flow_length_in_static = 0.0
        self.inlier_rate = 0.0
        self.foe = None
        self.foe_camstate_img = None
        self.tmp_moving_prob = None  # 0: inlier=stop, 1: outlier=moving
        self.moving_prob = None
        self.moving_prob_img = None
        self.intermediate_foe_img = None

    def compute(self, flow, sky_mask, nonsky_static_mask):
        """..ext.
        compute focus of expansion from optical flow.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
            sky_mask: mask of sky. shape = (height, width), dtype = bool.
            nonsky_static_mask: mask of static object except sky like grounds. shape = (height, width), dtype = bool
        """
        self.flow = flow
        self.sky_mask = sky_mask
        self.nonsky_static_mask = nonsky_static_mask

        self.prepare_variables()

        self.comp_flow_existing_rate_in_static()

        if self.flow_existing_rate_in_static < self.THRE_FLOW_EXISTING_RATE:
            self.state = CameraState.STOPPING
            # at this moment, all flow existing pixel inside non-static mask is set as moving.
            self.comp_flow_existence_in_nonstatic()
            self.moving_prob = self.tmp_moving_prob.copy()
        else:
            self.comp_foe_by_ransac()

    def draw(self, bg_img=None):
        self.bg_img = bg_img

        self.prepare_canvas()
        self.draw_state()
        self.draw_moving_prob()

    def prepare_variables(self):
        self.state = CameraState.ROTATING
        self.tmp_moving_prob = (
            np.ones((self.flow.shape[0], self.flow.shape[1]), dtype=np.float16) * 0.5
        )
        self.tmp_moving_prob[self.sky_mask == True] = 0.0
        self.moving_prob = self.tmp_moving_prob.copy()

    def comp_flow_existing_rate_in_static(self):
        num_flow_existing_pix_in_static = 0

        # check pixels inside static mask
        staticpix_indices = np.where(self.nonsky_static_mask == True)
        sum_flow_length = 0.0
        for i in range(len(staticpix_indices[0])):
            row = staticpix_indices[0][i]
            col = staticpix_indices[1][i]

            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            flow_length = np.sqrt(u**2 + v**2)
            sum_flow_length += flow_length
            if flow_length < self.THRE_FLOWLENGTH:
                self.tmp_moving_prob[row, col] = flow_length / self.THRE_FLOWLENGTH
            else:
                self.tmp_moving_prob[row, col] = 1.0
                num_flow_existing_pix_in_static += 1

        if len(staticpix_indices[0]) == 0:
            self.mean_flow_length_in_static = 0
            self.flow_existing_rate_in_static = 0
        else:
            self.mean_flow_length_in_static = sum_flow_length / len(
                staticpix_indices[0]
            )
            self.flow_existing_rate_in_static = num_flow_existing_pix_in_static / len(
                staticpix_indices[0]
            )

        if self.LOG_LEVEL > 0:
            print(
                f"[INFO] mean flow length in static: {self.mean_flow_length_in_static}"
            )
            print(
                f"[INFO] flow existing pixel rate: {self.flow_existing_rate_in_static * 100:.2f} %"
            )

    def comp_flow_existence_in_nonstatic(self):
        # check pixels inside non static mask
        nonstaticpix_indices = np.where(
            (self.nonsky_static_mask == False) & (self.sky_mask == False)
        )
        for i in range(len(nonstaticpix_indices[0])):
            row = nonstaticpix_indices[0][i]
            col = nonstaticpix_indices[1][i]

            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            flow_lentgh = np.sqrt(u**2 + v**2)
            if flow_lentgh < self.THRE_FLOWLENGTH:
                self.tmp_moving_prob[row, col] = flow_lentgh / self.THRE_FLOWLENGTH
            else:
                self.tmp_moving_prob[row, col] = 1.0

    def comp_foe_by_ransac(self):
        """
        compute FoE by RANSAC
        """
        # if we set below as 0.0, it caused a error in self.foe = self._comp_crosspt().
        max_inlier_rate = 1e-6
        self.foe = None
        self.inlier_foe2pt_mat = None
        if self.LOG_LEVEL > 3:
            # need to reset every time because it might be pressed 'q' to dkip the previous drawing image.
            self.display_foe_flow_img = True

        for _ in range(self.NUM_RANSAC):
            foe_candi = self.comp_foe_candidate()
            if foe_candi is None:
                continue
            if self.foe is None:
                # initialize by the first candidate
                self.foe = foe_candi

            self.comp_inlierrate_movpixprob(foe_candi)

            if self.inlier_rate > max_inlier_rate:
                # update by the current best
                max_inlier_rate = self.inlier_rate
                self.moving_prob = self.tmp_moving_prob.copy()
                self.inlier_foe2pt_mat = self.tmp_inlier_foe2pt_mat.copy()

                # stop if inlier rate is high enough
                if self.inlier_rate > self.THRE_INLIER_RATE:
                    self.state = CameraState.ONLY_TRANSLATING
                    break

        if self.RANSAC_ALL_INLIER_ESTIMATION:
            # currently this function is very slow and performance becomes lower, so it might be better to comment out this function.
            self.foe = self._comp_crosspt()

            if self.LOG_LEVEL > 2:
                # check distance from foe_candi to foe
                print(
                    f"[INFO] distance from foe_candi to foe [pix] = "
                    f"{np.linalg.norm(foe_candi[0:1]/foe_candi[2] - self.foe[0:1]/self.foe[2])}"
                )

    def comp_foe_candidate(self) -> np.ndarray:
        """
        compute FoE from 2 flow lines only inside static mask.
        FoE is in 3D homogeneous coordinate.
        """
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
        if self.LOG_LEVEL > 2:
            self.intermediate_foe_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8
            )
            self.draw_line(l1, self.intermediate_foe_img)
            self.draw_line(l2, self.intermediate_foe_img)
            self.draw_homogeneous_point(foe, self.intermediate_foe_img)
            cv2.imshow("Debug", self.intermediate_foe_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()

        return foe

    def comp_flowline(self, row: int, col: int) -> np.ndarray:
        """
        compute flow line from flow at (row, col)
        args:
            row: row index of flow
            col: column index of flow
        return:
            line: flow line in 3D homogeneous coordinate. if flow is too small, return None.
        """
        x = [col, row, 1]

        # flow
        u = self.flow[row, col, 0]
        v = self.flow[row, col, 1]

        # if flow is too small, return zero line
        flow_lentgh = np.sqrt(u**2 + v**2)
        if flow_lentgh < self.THRE_FLOWLENGTH:
            return None

        x_prev = [col - u, row - v, 1]

        if self.LOG_LEVEL > 2:
            cv2.arrowedLine(
                self.intermediate_foe_img,
                tuple(map(int, x_prev[0:2])),
                tuple(x[0:2]),
                (0, 0, 255),
                3,
            )

        # no rotation correction version
        line = np.cross(x, x_prev)
        return line

    def random_point_in_static_mask(self):
        # Find the indices of all pixels in the static mask that have a value of 1
        indices = np.where(self.nonsky_static_mask == 1)

        # Randomly select one of the indices
        index = np.random.choice(len(indices[0]))

        # Get the row and column values corresponding to the selected index
        row = indices[0][index]
        col = indices[1][index]

        return row, col

    def _comp_crosspt(self) -> np.ndarray:
        """
        Compute the most probable crossing point (Focus of Expansion) from the inlier lines.
        return:
            crossing_point: crossing point in 3D homogeneous coordinate.
        """
        # Check that there are enough lines to compute the crossing point
        if self.inlier_foe2pt_mat is None or self.inlier_foe2pt_mat.shape[0] < 2:
            print("[WARNING] Not enough lines to compute the crossing point.")
            return None

        # to avoid _ArrayMemoryError, use scipy's svd instead of np.linalg.svd.
        # Convert the matrix to a sparse representation
        sparse_matrix = sparse.csr_matrix(self.inlier_foe2pt_mat)
        # to avoid ValueError: `k` must be an integer satisfying `0 < k < min(A.shape)`"
        if min(sparse_matrix.shape) < 2:
            print(f"[WARNING] parse_matrix.shape = {sparse_matrix.shape} is too small.")
            return None
        U, S, Vt = sparse.linalg.svds(sparse_matrix, k=1)

        # The crossing point is the last column of V (or Vt.T[-1])
        crossing_point_homogeneous = Vt.T[:, -1]

        # Normalize the homogeneous coordinates
        crossing_point = crossing_point_homogeneous / crossing_point_homogeneous[-1]

        return crossing_point

    def comp_inlierrate_movpixprob(self, foe) -> float:
        """
        compute inlier rate except sky mask from FoE,
        and temporary moving pixel probability by length and angle difference,
        in the same pixel for loop.
        args:
            foe: FoE in 3D homogeneous coordinate
        """
        num_inlier = 0
        num_flow_existingpix = 0
        self.tmp_inlier_foe2pt_mat = None

        # treat candidate FoE is infinite case
        if foe[2] == 0:
            foe[2] = 1e-10
        else:
            foe_u = foe[0] / foe[2]
            foe_v = foe[1] / foe[2]

        # check pixels inside flow existing static mask area.
        for row, col in zip(
            *np.nonzero(self.tmp_moving_prob[:: self.SEARCH_STEP, :: self.SEARCH_STEP])
        ):
            # get flow
            flow_u = self.flow[row, col, 0]
            flow_v = self.flow[row, col, 1]

            flow_length = np.sqrt(flow_u**2 + flow_v**2)
            # TODO: I need to check whether this (tanh) definition is OK.
            # probably, e.g. 100 times difference should be more exaggerated than this.
            length_diff_prob = min(
                1.0,
                max(
                    np.tanh(abs(flow_length / self.mean_flow_length_in_static - 1)),
                    self.SAME_FLOWLENGTH_MIN_MOVING_PROB,
                ),
            )
            if flow_length < self.THRE_FLOWLENGTH:
                # this means that the nonstatic object moves with the camera.
                # And the camera is not stopping, so the object must be moving.
                self.tmp_moving_prob[row, col] = (
                    length_diff_prob * self.SAME_FLOWANGLE_MIN_MOVING_PROB
                )
            else:
                num_flow_existingpix += 1

                if self.LOG_LEVEL > 3 and self.display_foe_flow_img:
                    self._show_foe_flow_img(row, col, foe_u, foe_v, flow_u, flow_v)

                # check the angle between flow and FoE-to-each-pixel is lower than the threshold.
                foe2pt = np.array([col - foe_u, row - foe_v, 1])
                cos_foe_flow = np.dot((foe2pt[0], foe2pt[1]), (flow_u, flow_v)) / (
                    np.sqrt(foe2pt[0] ** 2 + foe2pt[1] ** 2) * flow_length
                )
                angle_diff_prob = min(
                    1.0, max(1 - cos_foe_flow, self.SAME_FLOWANGLE_MIN_MOVING_PROB)
                )
                self.tmp_moving_prob[row, col] = angle_diff_prob * length_diff_prob
                # count up inlier if the angle is lower than the threshold.
                if cos_foe_flow > self.THRE_COS_INLIER:
                    num_inlier += 1
                    # add inlier foe2pt vector to matrix as a row vector to compute FoE by RANSAC
                    if num_inlier == 1:
                        self.tmp_inlier_foe2pt_mat = foe2pt
                    else:
                        self.tmp_inlier_foe2pt_mat = np.vstack(
                            (self.tmp_inlier_foe2pt_mat, foe2pt)
                        )

        if self.flow_existing_rate_in_static == 0:
            self.inlier_rate = 0
        else:
            self.inlier_rate = num_inlier / num_flow_existingpix

        if self.LOG_LEVEL > 0:
            print(
                f"[INFO] FoE candidate: {foe}, inlier rate: {self.inlier_rate * 100:.2f} %"
            )

    def _show_foe_flow_img(self, row, col, foe_u, foe_v, flow_u, flow_v):
        LENGTH_FACTOR = 10
        foe_flow_img = self.intermediate_foe_img.copy()
        cv2.arrowedLine(
            foe_flow_img,
            (int(foe_u), int(foe_v)),
            (col, row),
            (0, 255, 0),
            3,
        )
        cv2.arrowedLine(
            foe_flow_img,
            (col, row),
            (int(col + flow_u * LENGTH_FACTOR), int(row + flow_v * LENGTH_FACTOR)),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            foe_flow_img,
            "you can close this image by pressing 'q'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("FoE and flow", foe_flow_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.display_foe_flow_img = False
            cv2.destroyWindow("FoE and flow")

    def prepare_canvas(self):
        if self.bg_img is None:
            self.foe_camstate_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8
            )
        else:
            self.foe_camstate_img = self.bg_img.copy()
            if self.LOG_LEVEL > 1:
                self.draw_flowarrow(self.flow, self.foe_camstate_img)
            if self.LOG_LEVEL > 2:
                self.intermediate_foe_img = self.bg_img.copy()

    def draw_flowarrow(self, flow, img):
        """
        draw flow as arrow
        """
        for row in range(0, flow.shape[0], self.FLOWARROW_STEP):
            for col in range(0, flow.shape[1], self.FLOWARROW_STEP):
                u = flow[row, col, 0]
                v = flow[row, col, 1]
                cv2.arrowedLine(
                    img,
                    pt1=(int(col - u), int(row - v)),
                    pt2=(col, row),
                    color=(0, 0, 255),
                    thickness=3,
                )

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

    def draw_state(self):
        if self.state == CameraState.STOPPING:
            cv2.putText(
                self.foe_camstate_img,
                "Camera is stopping",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
        elif self.state == CameraState.ONLY_TRANSLATING:
            self.draw_homogeneous_point(self.foe, self.foe_camstate_img)
            cv2.putText(
                self.foe_camstate_img,
                "Camera is only translating",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
        elif self.state == CameraState.ROTATING:
            cv2.putText(
                self.foe_camstate_img,
                "Camera is rotating",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

    def draw_moving_prob(self):
        if self.moving_prob is None:
            return
        self.moving_prob_img = cv2.applyColorMap(
            np.uint8(self.moving_prob * 255), cv2.COLORMAP_JET
        )
        cv2.putText(
            self.moving_prob_img,
            "FoE based likelihood",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
