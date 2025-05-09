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
        num_ransac=100,
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
        self.foe_sign = 1  # +1 for source (the flow is expanding, which means camera is toward front), -1 for sink
        self.foe_camstate_img = None
        self.moving_prob = None  # 0: inlier=stop, 1: outlier=moving
        self.moving_prob_img = None
        self.intermediate_foe_img = None

    def compute(self, flow, sky_mask, static_mask):
        """..ext.
        compute focus of expansion from optical flow.
        args:
            flow: optical flow. shape = (height, width, 2): 2 channel corresponds to (u, v)
            sky_mask: mask of sky. shape = (height, width), dtype = bool.
            static_mask: mask of static (very low prior moving probability segment) object except sky like grounds. shape = (height, width), dtype = bool
        """
        self.flow = flow
        self.sky_mask = sky_mask
        self.static_mask = static_mask

        self.prepare_variables()

        self.comp_flow_existing_rate_in_static()

        if self.flow_existing_rate_in_static < self.THRE_FLOW_EXISTING_RATE:
            self.state = CameraState.STOPPING
            self.comp_flow_existence()
        else:
            # camera is considered as moving (set as rotating by default)
            self.comp_foe_by_ransac()
            self.comp_movpixprob(self.foe)

    def draw(self, bg_img=None):
        self.bg_img = bg_img

        self.prepare_canvas()
        self.draw_state()
        self.draw_moving_prob()

    def prepare_variables(self):
        self.state = CameraState.ROTATING
        self.moving_prob = (
            np.ones((self.flow.shape[0], self.flow.shape[1]), dtype=np.float16) * 0.5
        )
        self.moving_prob[self.sky_mask == True] = 0.0
        self.foe = None
        self.foe_sign = 1

    def comp_flow_existing_rate_in_static(self):
        num_flow_existing_pix_in_static = 0

        # check pixels inside static mask
        staticpix_indices = np.where(self.static_mask == True)
        sum_flow_length = 0.0
        for i in range(len(staticpix_indices[0])):
            row = staticpix_indices[0][i]
            col = staticpix_indices[1][i]

            # get flow
            u = self.flow[row, col, 0]
            v = self.flow[row, col, 1]

            # count flow existing pix in static mask, and compute temporary moving prob.
            flow_length = np.sqrt(u**2 + v**2)
            sum_flow_length += flow_length
            if flow_length > self.THRE_FLOWLENGTH:
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

    def comp_flow_existence(self):
        # at this moment, all flow existing pixel will be set as moving.
        # check pixels inside non static mask
        for row in range(self.flow.shape[0]):
            for col in range(self.flow.shape[1]):
                # get flow
                u = self.flow[row, col, 0]
                v = self.flow[row, col, 1]

                flow_lentgh = np.sqrt(u**2 + v**2)
                self.moving_prob[row, col] = min(
                    1.0, abs(flow_lentgh / self.THRE_FLOWLENGTH)
                )

    def comp_foe_by_ransac(self):
        """
        compute FoE by RANSAC, determine it's sign.
        """
        # if we set below as 0.0, it caused a error in self.foe = self._comp_crosspt().
        self.inlier_rate = 1e-6
        self.foe = None
        best_inlier_mat = None
        if self.LOG_LEVEL > 3:
            # need to reset every time because it might be pressed 'q' to dkip the previous drawing image.
            self.display_foe_flow_img = True

        for try_num in range(self.NUM_RANSAC):
            foe_candi = self.comp_foe_candidate()
            if foe_candi is None:  # this will unlikely happen.
                print(
                    f"[WARNING] foe_candi is None in ransac {try_num} trial. @ comp_foe_by_ransac"
                )
                continue

            inlier_rate_candi, inlier_mat_candi = self.comp_inlier_rate(foe_candi)

            if inlier_rate_candi > self.inlier_rate:
                # update by the current best
                self.inlier_rate = inlier_rate_candi
                best_inlier_mat = inlier_mat_candi
                self.foe = foe_candi

                # stop if inlier rate is high enough
                if self.inlier_rate > self.THRE_INLIER_RATE:
                    self.state = CameraState.ONLY_TRANSLATING
                    if self.LOG_LEVEL > 0:
                        foe_candi_uvcoordi = foe_candi[0:2] / foe_candi[2]
                        print(
                            f"[INFO] RANSAC {try_num} trial: "
                            f"FoE candidate: {foe_candi_uvcoordi}, "
                            f"FoE candidate sign: {self.foe_sign}, "
                        )
                    break

        if self.RANSAC_ALL_INLIER_ESTIMATION and (best_inlier_mat.shape[0] > 1):
            # currently this function is very slow and performance becomes lower, so it might be better to comment out this function.
            refined_foe = self._comp_crosspt(best_inlier_mat)
            if refined_foe is not None:
                self.foe = refined_foe

            if self.LOG_LEVEL > 0:
                foe_uvcoordi = self.foe[0:2] / self.foe[2]
                # check distance from foe_candi to foe
                print(
                    f"[INFO] RANSAC all inlier estimation: "
                    f"FoE: {foe_uvcoordi}, "
                    f"FoE sign: {self.foe_sign}, "
                    f"distance from FoE_candi to FoE [pix] = "
                    f"{np.linalg.norm(foe_candi_uvcoordi - foe_uvcoordi)}"
                )

    def comp_foe_candidate(self) -> np.ndarray:
        """
        compute FoE from 2 flow lines only inside static mask.
        FoE is in 3D homogeneous coordinate.
        If candidate is None, reselect 2 flow lines.
        The camera is considered not stopping when this function is called, so there must exist 2 flow lines.
        return:
            foe_candi: candidate FoE in 3D homogeneous coordinate.
        """

        def _get_random_flowline(max_retries=1000) -> np.ndarray:
            """Pick a random pixel from the static mask until a valid flow line is found."""
            for _ in range(max_retries):
                row, col = self.random_point_in_static_mask()
                line = self.comp_flowline(row, col)
                if line is not None:
                    return line
            raise RuntimeError(
                "[ERROR] Failed to find a valid flow line after {max_retry} attempts @ comp_foe_candidate"
            )

        # Get two valid flow lines and compute their cross product as the candidate FoE.
        l1 = _get_random_flowline()
        l2 = _get_random_flowline()
        foe_candi = np.cross(l1, l2)

        # draw debug image
        if self.LOG_LEVEL > 2:
            self.intermediate_foe_img = np.zeros(
                (self.flow.shape[0], self.flow.shape[1], 3), dtype=np.uint8
            )
            self.draw_line(l1, self.intermediate_foe_img)
            self.draw_line(l2, self.intermediate_foe_img)
            self.draw_homogeneous_point(foe_candi, self.intermediate_foe_img)
            cv2.imshow("Debug", self.intermediate_foe_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()

        return foe_candi

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
        indices = np.where(self.static_mask == 1)

        # Randomly select one of the indices
        index = np.random.choice(len(indices[0]))

        # Get the row and column values corresponding to the selected index
        row = indices[0][index]
        col = indices[1][index]

        return row, col

    def _comp_crosspt(self, inlier_mat: np.ndarray) -> np.ndarray | None:
        """
        Compute the most probable crossing point (Focus of Expansion) from the inlier lines.
        Args:
            inlier_mat: 2D array of inlier lines. shape = (num_lines, 3)
        Returns:
            crossing_point: crossing point in 3D homogeneous coordinate.
        """
        # Check that there are enough lines to compute the crossing point
        if (inlier_mat is None) or (inlier_mat.ndim != 2) or (inlier_mat.shape[0] < 2):
            print("[WARNING] Not enough lines to compute the crossing point.")
            return None

        # to avoid _ArrayMemoryError, use scipy's svd instead of np.linalg.svd.
        # Convert the matrix to a sparse representation
        sparse_matrix = sparse.csr_matrix(inlier_mat)
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

    def comp_inlier_rate(self, foe: np.ndarray):
        """
        compute inlier rate except sky mask from a candidate FoE.
        args:
            foe: FoE in 3D homogeneous coordinate
        Returns:
            tuple: (inlier_rate, inlier_foe2pt_mat)
                   Returns (0, None) if no flow exists.
        """
        num_inlier = 0
        num_flow_existingpix = 0
        sum_inlier_cos = 0.0
        # TODO: FoE should be computed by flow. not foe2pt. I have to fix this.
        inlier_foe2pt_mat = None

        # treat candidate FoE is infinite case
        if foe[2] == 0:
            foe[2] = 1e-10

        foe_u = foe[0] / foe[2]
        foe_v = foe[1] / foe[2]

        # check pixels inside flow existing static mask area.
        for row in range(0, self.static_mask.shape[0], self.SEARCH_STEP):
            for col in range(0, self.static_mask.shape[1], self.SEARCH_STEP):
                if self.static_mask[row, col]:
                    # get flow
                    flow_u = self.flow[row, col, 0]
                    flow_v = self.flow[row, col, 1]
                    flow_length = np.sqrt(flow_u**2 + flow_v**2)

                    if flow_length > self.THRE_FLOWLENGTH:
                        num_flow_existingpix += 1

                        # check the angle between flow and FoE-to-each-pixel is lower than the threshold.
                        foe2pt = np.array([col - foe_u, row - foe_v, 1])
                        foe2pt_length = np.sqrt(foe2pt[0] ** 2 + foe2pt[1] ** 2)
                        foe2pt_length = max(
                            foe2pt_length, 1e-6
                        )  # avoid division by zero
                        cos_foe_flow = np.dot(
                            (foe2pt[0], foe2pt[1]), (flow_u, flow_v)
                        ) / (foe2pt_length * flow_length)
                        cos_foe_flow = np.clip(cos_foe_flow, -1.0, 1.0)

                        # count up as an inlier when the angle is close.
                        if cos_foe_flow > self.THRE_COS_INLIER:
                            num_inlier += 1
                            sum_inlier_cos += cos_foe_flow
                            # add inlier foe2pt vector to matrix as a row vector to compute FoE by RANSAC
                            if inlier_foe2pt_mat is None:
                                inlier_foe2pt_mat = foe2pt
                            else:
                                inlier_foe2pt_mat = np.vstack(
                                    (inlier_foe2pt_mat, foe2pt)
                                )

                        if self.LOG_LEVEL > 3 and self.display_foe_flow_img:
                            self._show_foe_flow_img(
                                row, col, foe_u, foe_v, flow_u, flow_v, cos_foe_flow
                            )

        # Calculate results
        inlier_rate = 0.0
        mean_inlier_cos = 0.0
        if num_flow_existingpix > 0:
            inlier_rate = num_inlier / num_flow_existingpix
            if num_inlier > 0:
                mean_inlier_cos = sum_inlier_cos / num_inlier

        if mean_inlier_cos < 0:
            # if mean_inlier_cos is negative, it means the flow is opposite direction.
            # so we need to change the sign of foe; default = 1.
            self.foe_sign = -1

        if self.LOG_LEVEL > 0:
            foe_uvcoordi = foe[0:2] / foe[2]
            print(
                f"[INFO] FoE candidate: {foe_uvcoordi}, inlier rate: {inlier_rate * 100:.2f} %,"
                f" mean inlier cos: {mean_inlier_cos:.2f}"
            )

        return inlier_rate, inlier_foe2pt_mat

    def comp_movpixprob(self, foe) -> float:
        """
        compute moving pixel probability by length and angle difference.
        args:
            foe: FoE in 3D homogeneous coordinate
        """
        if foe is None:
            raise ValueError("FoE is None @ comp_movpixprob")

        # treat candidate FoE is infinite case
        if abs(foe[2]) < 1e-10:
            foe[2] = 1e-10
        foe_u = foe[0] / foe[2]
        foe_v = foe[1] / foe[2]

        self.mean_flow_length_in_static = max(self.mean_flow_length_in_static, 1e-10)

        for row in range(0, self.flow.shape[0]):
            for col in range(0, self.flow.shape[1]):
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
                    # Currently the camera is judged as moving in the former process,
                    # so this means that the former considered static object moves with the camera.
                    self.moving_prob[row, col] = (
                        length_diff_prob * self.SAME_FLOWANGLE_MIN_MOVING_PROB
                    )
                else:
                    # check the angle between flow and expected flow direction (signed FoE-to-each-pixel) is lower than the threshold.
                    expect_flowdir = self.foe_sign * np.array(
                        [col - foe_u, row - foe_v, 1]
                    )
                    expect_flowdir_length = np.sqrt(
                        expect_flowdir[0] ** 2 + expect_flowdir[1] ** 2
                    )
                    if expect_flowdir_length < 1e-6:  # avoid division by zero
                        expect_flowdir_length = 1e-6
                    cos_foe_flow = np.dot(
                        (expect_flowdir[0], expect_flowdir[1]), (flow_u, flow_v)
                    ) / (expect_flowdir_length * flow_length)
                    angle_diff_prob = min(
                        1.0,
                        max(
                            (1 - cos_foe_flow) / 2.0,
                            self.SAME_FLOWANGLE_MIN_MOVING_PROB,
                        ),
                    )
                    self.moving_prob[row, col] = angle_diff_prob * length_diff_prob

        # Ensure sky mask remains 0 probability after calculations
        self.moving_prob[self.sky_mask == True] = 0.0

    def _show_foe_flow_img(self, row, col, foe_u, foe_v, flow_u, flow_v, cos_foe_flow):
        LENGTH_FACTOR = 10
        foe_flow_img = self.intermediate_foe_img.copy()

        # paint self.static_mask area as gray
        foe_flow_img[self.static_mask] = [128, 128, 128]

        # draw arrows.
        if cos_foe_flow > 0:
            cv2.arrowedLine(
                foe_flow_img,
                (int(foe_u), int(foe_v)),
                (col, row),
                (0, 255, 0),
                3,
            )
        else:
            # FoE sign is negative, so the flow direction is point to FoE.
            cv2.arrowedLine(
                foe_flow_img,
                (col, row),
                (int(foe_u), int(foe_v)),
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
                (255, 255, 255),
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
                (255, 255, 255),
                2,
            )
        elif self.state == CameraState.ROTATING:
            cv2.putText(
                self.foe_camstate_img,
                "Camera is rotating",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
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
