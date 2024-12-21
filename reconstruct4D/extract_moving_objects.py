from focus_of_expansion import FoE
import opticalflow
import segmentator
import argparse
import cv2
import os
import numpy as np


class MovingObjectExtractor:
    def __init__(self, args) -> None:
        # constants
        self.RESULTIMG_WIDTH = args.resultimg_width
        # if moving probability is lower than this value, the pixel is considered as static. default value = prior(0.5) * angle likelihood(0.5) * length likelihood(0.5).
        self.THRE_MOVING_PROB = 0.5**3
        THRE_STATIC_PROB = 0.1
        THRE_DOMINANTFLOW_ANGLE = 10 * np.pi / 180
        # if flow length is lower than this value, the flow orientation will be ignored.
        THRE_FLOWLENGTH = 0.2
        # if angle between flow and foe-pos is lower than this value, the flow considered as an inlier.[radian]
        THRE_INLIER_ANGLE = 10 * np.pi / 180
        # if inlier rate is higher than this value, RANSAC will be stopped.
        THRE_INLIER_RATE = 0.9
        # if flow existing pixel rate is lower than this value, the camera is considered as stopping.
        # the flow existing rate will be computed only inside static mask.
        THRE_FLOW_EXISTING_RATE = 0.01
        NUM_RANSAC = 10
        # every this pixel, draw flow arrow.
        FLOWARROW_STEP = 20
        # minimum moving probability even when the angle is totally the same with FoE-position angle, or the flow length is the same with background.
        SAME_FLOWANGLE_MOVING_PROB = 0.2
        # minimum moving probability even when the flow length is the same with background.
        SAME_FLOWLENGTH_MOVING_PROB = 0.4
        # in moving pixel region, it is considered as moving object if the same object id exists over this rate.
        self.THRE_MOVINGOBJ_AREA_RATE = 0.2

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
            THRE_FLOWLENGTH, THRE_DOMINANTFLOW_ANGLE, args.loglevel
        )
        # Segmentator initialization based on the model type
        if args.segment_model_type == "internimage":
            self.seg = segmentator.InternImageSegmentatorWrapper(
                model_name=None,
                input_dir=None,
                result_dir=args.segment_result_dir,
                thre_static_prob=THRE_STATIC_PROB,
                log_level=args.loglevel,
            )
        elif args.segment_model_type == "oneformer":
            self.seg = segmentator.OneFormerSegmentatorWrapper(
                model_name=args.segment_model_name,
                task_type=args.segment_task_type,
                input_dir=args.input_dir,
                result_dir=args.segment_result_dir,
                thre_static_prob=THRE_STATIC_PROB,
                log_level=args.loglevel,
            )
        else:
            print(f"[ERROR] unknown segment model type: {args.segment_model_type}")
            self.seg = None
        self.seg.load_prior()
        self.foe = FoE(
            THRE_FLOWLENGTH,
            THRE_INLIER_ANGLE,
            THRE_INLIER_RATE,
            THRE_FLOW_EXISTING_RATE,
            NUM_RANSAC,
            SAME_FLOWANGLE_MOVING_PROB,
            SAME_FLOWLENGTH_MOVING_PROB,
            FLOWARROW_STEP,
            log_level=args.loglevel,
        )
        self.cur_imgname = None
        self.cur_img = None
        self.posterior_movpix_prob = None

    def compute(self):
        # process each image
        for self.cur_imgname in self.imgfiles:
            # skip frames at the beginning if it is specified.
            if args.skip_frames > 0:
                args.skip_frames -= 1
                continue

            self.process_image()
            self.draw()

        cv2.destroyAllWindows()

    def process_image(self):
        self.cur_img = cv2.imread(os.path.join(args.input_dir, self.cur_imgname))

        if args.loglevel > 0:
            print(f"[INFO] processing {self.cur_imgname} : {self.cur_img.shape}")

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow.
        # That is different from what we expect, which is t-1 to t flow.
        # So in the future, we need to compute flow from t-1 to t and overlap it with the current image.
        self.optflow.compute(self.cur_imgname)

        # currently just read segmentation result from corresponding image file name.
        self.seg.compute(self.cur_imgname)

        # compute focus of expansion
        self.foe.compute(
            self.optflow.flow, self.seg.sky_mask, self.seg.nonsky_static_mask
        )

        # compute posterior probability of moving pixels
        self.posterior_movpix_prob = self.seg.moving_prob * self.foe.moving_prob
        self.posterior_movpix_mask = self.posterior_movpix_prob > self.THRE_MOVING_PROB

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

        self._write_allimgtitles()

        # combine intermediate images
        self.seg.draw(bg_img=self.cur_img)
        self.foe.draw(bg_img=self.optflow.flow_img)
        row1_img = cv2.hconcat(
            [self.cur_img, self.seg.result_img, self.seg.result_movingmask_img]
        )
        row2_img = cv2.hconcat(
            [self.seg.moving_prob_img, self.optflow.flow_img, self.foe.foe_camstate_img]
        )
        row3_img = cv2.hconcat(
            [self.foe.moving_prob_img, self.posterior_movpix_prob_img, self.result_img]
        )
        result_comb_img = cv2.vconcat([row1_img, row2_img, row3_img])
        # resize keeping result image aspect ratio
        comb_imgsize = (
            self.RESULTIMG_WIDTH,
            int(
                self.RESULTIMG_WIDTH
                * self.result_img.shape[0]
                / self.result_img.shape[1]
            ),
        )
        result_comb_img = cv2.resize(result_comb_img, comb_imgsize)

        if args.loglevel > 0:
            print(f"imgshape={self.cur_img.shape}")
            print(f"comb_imgsize={comb_imgsize}")

        # display the result image
        if args.loglevel > 1:
            cv2.imshow("result", result_comb_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                return

        # create the result image and save it.
        # save segmentation image if it is not saved yet.
        if not os.path.exists(f"{args.segment_result_dir}/{self.cur_imgname}"):
            cv2.imwrite(
                f"{args.segment_result_dir}/{self.cur_imgname}", self.seg.result_img
            )

        base_name = os.path.splitext(self.cur_imgname)[0]
        save_imgname = f"{base_name}_result.png"
        cv2.imwrite(f"{args.output_dir}/{save_imgname}", self.result_img)
        save_comb_imgname = f"{base_name}_result_comb.png"
        cv2.imwrite(f"{args.output_dir}/{save_comb_imgname}", result_comb_img)

        # save the posterior mask image
        # the mask value should be 0 or 255 becuase it will be automatically /255 in evaluation time.
        posterior_mask_img = self.moving_obj_mask * 255
        posterior_mask_imgfname = f"{args.output_dir}/{base_name}_mask.png"
        cv2.imwrite(posterior_mask_imgfname, posterior_mask_img)

        if args.loglevel > 2:
            # check loaded image type.
            loaded_posterior_mask_img = cv2.imread(
                posterior_mask_imgfname, cv2.IMREAD_UNCHANGED
            )
            print(f"loaded_mask_img.shape={loaded_posterior_mask_img.shape}")
            print(f"loaded_mask_img.dtype={loaded_posterior_mask_img.dtype}")
            cv2.imshow("loaded_mask_img", loaded_posterior_mask_img)
            cv2.waitKey(1)

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
        self._write_imgtitle(self.result_img, "result")

    def _compute_moving_obj_mask(self):
        """
        Compute moving object mask based on the posterior probability of moving pixels.
        Moving object is defined as segmented object which has some moving pixel area. The detail definition is below.
        In a moving pixels connected region, if the same object id exists over thresold are percentage, the object is considered as moving object.
        object id can get from seg.id_mask, which is torch.tensor.
        """
        self.moving_obj_mask = np.zeros_like(self.posterior_movpix_mask, dtype=np.uint8)
        # find connected regions of moving pixels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.posterior_movpix_mask.astype(np.uint8), connectivity=8
        )
        # compute moving object mask
        for label in range(1, num_labels):
            # extract current label area mask
            label_mask = labels == label
            label_area = stats[label, cv2.CC_STAT_AREA]
            # get object id in the label region
            obj_ids, obj_areas = np.unique(
                self.seg.id_mask[label_mask], return_counts=True
            )
            # get the object id which has over threshold area in the label region.
            dominant_obj_ids = obj_ids[
                obj_areas > self.THRE_MOVINGOBJ_AREA_RATE * label_area
            ]
            # update moving object mask
            self.moving_obj_mask = (
                np.isin(self.seg.id_mask, dominant_obj_ids) | self.moving_obj_mask
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract moving objects")
    parser.add_argument(
        "--input_dir", type=str, default="../data/sample", help="input image directory"
    )
    parser.add_argument(
        "--flow_result_dir",
        type=str,
        default="../output/sample/flow",
        help="optical flow result directory",
    )
    parser.add_argument(
        "--segment_model_type",
        type=str,
        default="internimage",
        help="segmentation model type: internimage or oneformer",
    )
    parser.add_argument(
        "--segment_model_name", type=str, default=None, help="segmentation model name"
    )
    parser.add_argument(
        "--segment_task_type",
        type=str,
        default="panoptic",
        help="segmentation task type: panoptic or semantic or instance",
    )
    parser.add_argument(
        "--segment_result_dir",
        type=str,
        default="../output/sample/segment",
        help="segmentation result directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/sample/final",
        help="output image directory",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=3,
        help="log level:0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image",
    )
    parser.add_argument(
        "--resultimg_width", type=int, default=1280, help="result image width.[pix]"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=0, help="skip frames at the beginning"
    )
    args = parser.parse_args()

    print("[INFO] Parameters passed to MovingObjectExtractor:")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    moe = MovingObjectExtractor(args)
    moe.compute()
