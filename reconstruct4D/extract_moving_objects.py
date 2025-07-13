import argparse
import logging
import os
import time
import yaml 

import cv2
import numpy as np
import opticalflow
import segmentator
from focus_of_expansion import FoE


class MovingObjectExtractor:
    def __init__(self, args) -> None:
        # constants from args
        self.args = args
        self.logger = logging.getLogger(__name__)

        # variables
        self.imgfiles = sorted(
            [
                file
                for file in os.listdir(args.input_dir)
                if file.endswith(".jpg") or file.endswith(".png")
            ]
        )
        # remove last frame because currently optical flow is computed from t and t+1, so there is no optical flow file for the last frame.
        self.imgfiles.pop()
        print(f"[INFO] reading input image files: {self.imgfiles}")
        self.optflow = opticalflow.UnimatchFlow(args.flow_result_dir)
        self.undominantflow = opticalflow.UndominantFlowAngleExtractor(
            args.thre_flowlength, args.thre_dominantflow_angle, args.loglevel
        )
        # Segmentator initialization based on the model type
        if args.segment_model_type == "internimage":
            self.seg = segmentator.InternImageSegmentatorWrapper(
                model_name=None,
                input_dir=None,
                result_dir=args.segment_result_dir,
                thre_static_prob=args.thre_static_prob,
                log_level=args.loglevel,
            )
        elif args.segment_model_type == "oneformer":
            self.seg = segmentator.OneFormerSegmentatorWrapper(
                model_name=args.segment_model_name,
                task_type=args.segment_task_type,
                input_dir=args.input_dir,
                result_dir=args.segment_result_dir,
                thre_static_prob=args.thre_static_prob,
                log_level=args.loglevel,
            )
        else:
            print(f"[ERROR] unknown segment model type: {args.segment_model_type}")
            self.seg = None
        self.seg.load_prior()
        self.foe = FoE(
            args.thre_flowlength,
            args.thre_inlier_angle,
            args.thre_inlier_rate,
            args.thre_flow_existing_rate,
            args.num_ransac,
            args.flowarrow_step_forvis,
            args.flowlength_factor_forvis,
            args.ransac_all_inlier_estimation,
            args.foe_search_step,
            log_level=args.loglevel,
            movprob_lengthfactor_coeff=args.movprob_lengthfactor_coeff,
            thre_movprob_deg=args.thre_movprob_deg,
        )
        self.cur_imgname = None
        self.cur_img = None
        self.posterior_movpix_prob = None

    def compute(self):
        # process each image
        for self.cur_imgname in self.imgfiles:
            start_time = time.time()

            # skip frames at the beginning if it is specified.
            if self.args.skip_frames > 0:
                self.args.skip_frames -= 1
                continue

            # skip if the result image is already saved.
            base_imgname = os.path.splitext(self.cur_imgname)[0]
            self.fullpath_result_imgname = (
                f"{self.args.result_dir}/{base_imgname}_result.png"
            )
            if os.path.exists(self.fullpath_result_imgname):
                self.logger.info(
                    f"[INFO] {self.fullpath_result_imgname} already exists. skipping."
                )
                continue

            self.process_image()
            process_time = time.time()
            self.draw()

            end_time = time.time()
            if self.args.loglevel > 0:
                print(f"[INFO] processing time = {process_time - start_time:.2f} [sec]")
                print(f"[INFO] drawing time = {end_time - process_time:.2f} [sec]")
                print(f"[INFO] total time = {end_time - start_time:.2f} [sec]")

        cv2.destroyAllWindows()

    def process_image(self):
        self.cur_img = cv2.imread(os.path.join(self.args.input_dir, self.cur_imgname))

        if self.args.loglevel > 0:
            print(
                f"------------\n[INFO] processing {self.args.input_dir}/{self.cur_imgname} : {self.cur_img.shape}"
            )

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow.
        # That is different from what we expect, which is t-1 to t flow.
        # So in the future, we need to compute flow from t-1 to t and overlap it with the current image.
        self.optflow.compute(self.cur_imgname)

        # currently just read segmentation result from corresponding image file name.
        self.seg.compute(self.cur_imgname)

        # compute camera is moving or not and focus of expansion
        self.foe.compute(self.optflow.flow, self.seg.sky_mask, self.seg.static_mask)

        # compute posterior probability of moving pixels
        self.posterior_movpix_prob = self.seg.moving_prob * self.foe.moving_prob
        self.posterior_movpix_mask = self.posterior_movpix_prob > self.args.thre_moving_prob

        self._compute_moving_obj_mask()

    def draw(self) -> None:
        if self.posterior_movpix_prob is None:
            return

        self.posterior_movpix_prob_img = cv2.applyColorMap(
            np.uint8(self.posterior_movpix_prob * 255), cv2.COLORMAP_JET
        )

        # overlay transparently outlier_mask(moving object mask) into input image
        overlay_img = self.cur_img.copy() // 2
        # increase the red channel.
        overlay_img[self.moving_obj_mask == 1, 2] += 128
        self.result_img = overlay_img

        # combine intermediate images
        self.seg.draw(bg_img=self.cur_img)
        self.foe.draw(bg_img=self.optflow.flow_img)
        movobj_mask_img = self.moving_obj_mask * 255
        self.movobj_mask_color_img = cv2.cvtColor(movobj_mask_img, cv2.COLOR_GRAY2BGR)

        self._write_allimgtitles()
        row1_img = cv2.hconcat(
            [self.cur_img, self.seg.result_img, self.seg.moving_prob_img]
        )
        row2_img = cv2.hconcat(
            [self.optflow.flow_img, self.foe.foe_camstate_img, self.foe.moving_prob_img]
        )
        row3_img = cv2.hconcat(
            [
                self.posterior_movpix_prob_img,
                self.movobj_mask_color_img,
                self.result_img,
            ]
        )
        result_comb_img = cv2.vconcat([row1_img, row2_img, row3_img])
        # resize keeping result image aspect ratio
        comb_imgsize = (
            self.args.resultimg_width,
            int(
                self.args.resultimg_width
                * self.result_img.shape[0]
                / self.result_img.shape[1]
            ),
        )
        result_comb_img = cv2.resize(result_comb_img, comb_imgsize)

        if self.args.loglevel > 0:
            print(f"imgshape={self.cur_img.shape}")
            print(f"comb_imgsize={comb_imgsize}")

        # display the result image
        if self.args.loglevel > 1:
            cv2.imshow("result", result_comb_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                return

        # create the result image and save it.
        # save segmentation image if it is not saved yet.
        if not os.path.exists(f"{self.args.segment_result_dir}/{self.cur_imgname}"):
            cv2.imwrite(
                f"{self.args.segment_result_dir}/{self.cur_imgname}", self.seg.result_img
            )

        # save the posterior mask image
        # the mask value should be 0 or 255 becuase it will be automatically /255 in evaluation time.
        base_imgname = os.path.splitext(self.cur_imgname)[0]
        movobj_mask_imgfname = f"{self.args.result_dir}/{base_imgname}_mask.png"
        cv2.imwrite(movobj_mask_imgfname, movobj_mask_img)

        cv2.imwrite(self.fullpath_result_imgname, self.result_img)
        save_comb_imgname = f"{base_imgname}_result_comb.png"
        cv2.imwrite(f"{self.args.result_dir}/{save_comb_imgname}", result_comb_img)

    def _write_imgtitle(self, img, caption, color=(255, 255, 255)):
        cv2.putText(
            img,
            caption,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    def _write_allimgtitles(self):
        self._write_imgtitle(self.cur_img, "input")
        self._write_imgtitle(self.seg.result_img, "segmentation")
        self._write_imgtitle(self.optflow.flow_img, "optical flow")
        self._write_imgtitle(self.posterior_movpix_prob_img, "posterior")
        self._write_imgtitle(self.movobj_mask_color_img, "moving_object_mask")
        # it might be better not to write "result" in final result image to display clean result.
        # self._write_imgtitle(self.result_img, "result")

    def _compute_moving_obj_mask(self):
        """
        Compute moving object mask.
        An object (defined by self.seg.id_mask) is considered moving if a sufficient fraction
        of its total pixels are determined to be moving by self.posterior_movpix_mask.
        """
        self.moving_obj_mask = np.zeros_like(self.posterior_movpix_mask, dtype=np.uint8)
        unique_seg_ids = np.unique(self.seg.id_mask)

        for seg_id in unique_seg_ids:
            current_seg_mask = np.asarray(self.seg.id_mask) == seg_id
            total_area = np.sum(current_seg_mask)
            if total_area == 0:
                continue
            moving_area = np.sum(current_seg_mask & self.posterior_movpix_mask)
            moving_area_rate = moving_area / total_area
            if moving_area_rate > self.args.thre_moving_fraction_in_obj:
                self.moving_obj_mask[current_seg_mask] = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract moving objects")
    
    parser.add_argument(
        "--config",
        type=str,
        default="../script/foels_param.yaml",
        help="path to the YAML config file",
    )

    temp_args, _ = parser.parse_known_args()

    config = {}
    if os.path.exists(temp_args.config):
        with open(temp_args.config, 'r') as f:
            config = yaml.safe_load(f).get('MovingObjectExtractor', {})

    parser = argparse.ArgumentParser(description="extract moving objects")

    parser.add_argument(
        "--input_dir", type=str, default=config.get("input_dir", "../data/sample"), help="input image directory"
    )
    parser.add_argument(
        "--flow_result_dir",
        type=str,
        default=config.get("flow_result_dir", "../result/sample/flow"),
        help="optical flow result directory",
    )
    parser.add_argument(
        "--segment_model_type",
        type=str,
        default=config.get("segment_model_type", "internimage"),
        help="segmentation model type: internimage or oneformer",
    )
    parser.add_argument(
        "--segment_model_name", type=str, default=config.get("segment_model_name"), help="segmentation model name"
    )
    parser.add_argument(
        "--segment_task_type",
        type=str,
        default=config.get("segment_task_type", "panoptic"),
        help="segmentation task type: panoptic or semantic or instance",
    )
    parser.add_argument(
        "--segment_result_dir",
        type=str,
        default=config.get("segment_result_dir", "../result/sample/segment"),
        help="segmentation result directory",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=config.get("result_dir", "../result/sample/final"),
        help="result image directory",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=config.get("loglevel", 3),
        help="log level:0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image",
    )
    parser.add_argument(
        "--resultimg_width", type=int, default=config.get("resultimg_width", 1280), help="result image width.[pix]"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=config.get("skip_frames", 0), help="skip frames at the beginning"
    )
    parser.add_argument(
        "--ransac_all_inlier_estimation",
        type=bool,
        default=config.get("ransac_all_inlier_estimation", False),
        help="whether all inlier estimation at the RANSAC final step.",
    )
    parser.add_argument(
        "--foe_search_step",
        type=int,
        default=config.get("foe_search_step", 1),
        help="search step size when computing FoE inlier",
    )
    parser.add_argument(
        "--num_ransac",
        type=int,
        default=config.get("num_ransac", 30),
        help="number of RANSAC iterations for FoE estimation",
    )
    parser.add_argument(
        "--thre_moving_fraction_in_obj",
        type=float,
        default=config.get("thre_moving_fraction_in_obj", 0.1),
        help="threshold of moving pixel fraction in an object to be considered as moving",
    )
    parser.add_argument(
        "--movprob_lengthfactor_coeff",
        type=float,
        default=config.get("movprob_lengthfactor_coeff", 0.25),
        help="length factor coefficient for moving probability. e.g. if the target flow length is 100 times compared with mean of static background flow, factor is 2, and moving probability will be increase to 2* this coeff ",
    )
    parser.add_argument(
        "--thre_movprob_deg",
        type=float,
        default=config.get("thre_movprob_deg", 30),
        help="threshold of moving probability of angle difference between flow and foe-pos to be considered as moving probabiity =0.5",
    )
    parser.add_argument(
        "--thre_moving_prob",
        type=float,
        default=config.get("thre_moving_prob", 0.25),
        help="if moving probability is lower than this value, the pixel is considered as static. default value = prior(0.5) * angle likelihood(0.5) * length likelihood(0.5).",
    )
    parser.add_argument(
        "--thre_static_prob",
        type=float,
        default=config.get("thre_static_prob", 0.1),
        help="threshold of static probability",
    )
    parser.add_argument(
        "--thre_dominantflow_angle",
        type=float,
        default=config.get("thre_dominantflow_angle", 0.17453292519943295),
        help="if flow angle is lower than this value, the flow is ignored. [radian]",
    )
    parser.add_argument(
        "--thre_flowlength",
        type=float,
        default=config.get("thre_flowlength", 1.0),
        help="if flow length is lower than this value, the flow orientation will be ignored.",
    )
    parser.add_argument(
        "--thre_inlier_angle",
        type=float,
        default=config.get("thre_inlier_angle", 0.03490658503988659),
        help="if angle between flow and foe-pos is lower than this value, the flow considered as an inlier.[radian]",
    )
    parser.add_argument(
        "--thre_inlier_rate",
        type=float,
        default=config.get("thre_inlier_rate", 0.6),
        help="if inlier rate is higher than this value, RANSAC will be stopped.",
    )
    parser.add_argument(
        "--thre_flow_existing_rate",
        type=float,
        default=config.get("thre_flow_existing_rate", 0.01),
        help="if flow existing pixel rate is lower than this value, the camera is considered as stopping.",
    )
    parser.add_argument(
        "--flowarrow_step_forvis",
        type=int,
        default=config.get("flowarrow_step_forvis", 20),
        help="every this pixel, draw flow arrow.",
    )
    parser.add_argument(
        "--flowlength_factor_forvis",
        type=int,
        default=config.get("flowlength_factor_forvis", 1),
        help="flow arrow length factor for visualization",
    )
    
    # 最終的に引数をパース
    args = parser.parse_args()


    print("[INFO] Parameters passed to MovingObjectExtractor:")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    moe = MovingObjectExtractor(args)
    moe.compute()