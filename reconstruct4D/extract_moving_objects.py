from focus_of_expansion import CameraState
from focus_of_expansion import FoE
import opticalflow
import segmentator
import argparse
import cv2
import os
import sys
import numpy as np


class MovingObjectExtractor:
    def __init__(self, args) -> None:
        # constants
        self.RESULTIMG_WIDTH = args.resultimg_width
        # if moving probability is lower than this value, the pixel is considered as static. default value = prior(0.5) * angle likelihood(0.5) * length likelihood(0.5).
        self.THRE_MOVING_PROB = 0.5**3
        THRE_STATIC_PROB = 0.1
        THRE_DOMINANTFLOW_ANGLE = 10*np.pi/180
        # if flow length is lower than this value, the flow orientation will be ignored.
        THRE_FLOWLENGTH = 0.2
        # if angle between flow and foe-pos is lower than this value, the flow considered as an inlier.[radian]
        THRE_INLIER_ANGLE = 10*np.pi/180
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

        # variables
        self.imgfiles = sorted([file for file in os.listdir(
            args.input_dir) if file.endswith('.jpg') or file.endswith('.png')])
        # remove last frame because currently optical flow is computed from t and t+1, so there is no optical flow file for the last frame.
        self.imgfiles.pop()
        print(f"[INFO] reading input image files: {self.imgfiles}")
        self.optflow = opticalflow.UnimatchFlow(args.flow_result_dir)
        self.undominantflow = opticalflow.UndominantFlowAngleExtractor(
            THRE_FLOWLENGTH, THRE_DOMINANTFLOW_ANGLE, args.loglevel)
        self.seg = segmentator.InternImageSegmentator(
            args.segment_result_dir, THRE_STATIC_PROB, args.loglevel)
        self.foe = FoE(THRE_FLOWLENGTH,THRE_INLIER_ANGLE,THRE_INLIER_RATE,THRE_FLOW_EXISTING_RATE,
                    NUM_RANSAC, SAME_FLOWANGLE_MOVING_PROB, SAME_FLOWLENGTH_MOVING_PROB, FLOWARROW_STEP, log_level=args.loglevel)
        self.cur_imgname = None
        self.cur_img = None
        self.posterior_moving_prob = None

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
        self.cur_img = cv2.imread(
            os.path.join(args.input_dir, self.cur_imgname))

        if args.loglevel > 0:
            print(
                f"[INFO] processing {self.cur_imgname} : {self.cur_img.shape}")

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow.
        # That is different from what we expect, which is t-1 to t flow.
        # So in the future, we need to compute flow from t-1 to t and overlap it with the current image.
        self.optflow.compute(self.cur_imgname)

        # currently jusr read regmentation result from corresponding image file name.
        self.seg.compute(self.cur_imgname)

        # compute focus of expansion
        self.foe.compute(self.optflow.flow,
                         self.seg.sky_mask, self.seg.nonsky_static_mask)

        # # stopping erea is defined as foe.inlier_mask[row, col] = 0
        # if self.foe.state == CameraState.ROTATING: 
        #     self.undominantflow.compute(
        #         self.optflow.flow, self.seg.nonsky_static_mask)
        #     self.foe.moving_prob = self.undominantflow.undominant_flow_prob

        # compute posterior probability of moving objects
        self.posterior_moving_prob = self.seg.moving_prob * self.foe.moving_prob

    def draw(self) -> None:
        if self.posterior_moving_prob is None:
            return
        
        posterior_moving_prob_img = cv2.applyColorMap(
            np.uint8(self.posterior_moving_prob * 255), cv2.COLORMAP_JET)
        cv2.putText(posterior_moving_prob_img, "posterior",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # overlay transparently outlier_mask(moving object mask) into input image
        overlay_img = self.cur_img.copy()//2
        # increase the red channel.
        overlay_img[self.posterior_moving_prob > self.THRE_MOVING_PROB, 2] += 128
        result_img = overlay_img

        if args.loglevel > 1:
            # display the result
            self.seg.draw(bg_img=self.cur_img)
            self.foe.draw(bg_img=self.optflow.flow_img)

            row1_img = cv2.hconcat(
                [self.cur_img, self.seg.result_img, self.seg.result_movingmask_img])
            row2_img = cv2.hconcat(
                [self.seg.moving_prob_img, self.optflow.flow_img, self.foe.foe_camstate_img])
            row3_img = cv2.hconcat(
                [self.foe.moving_prob_img, posterior_moving_prob_img, result_img])
            result_img = cv2.vconcat([row1_img, row2_img, row3_img])
            # resize keeping combined image aspect ratio
            save_imgsize = (self.RESULTIMG_WIDTH, int(
                self.RESULTIMG_WIDTH*result_img.shape[0]/result_img.shape[1]))
            print(f"imgshape={self.cur_img.shape}")
            print(f"save_imgsize={save_imgsize}")
            result_img = cv2.resize(
                result_img, save_imgsize)
            cv2.imshow('result', result_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                return

        # create the result image and save it.
        # change file extension to png
        save_imgname = self.cur_imgname.replace('.jpg', '.png')
        cv2.imwrite(f"{args.output_dir}/{save_imgname}", result_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract moving objects')
    parser.add_argument('--input_dir', type=str,
                        default='../data/sample', help='input image directory')
    parser.add_argument('--flow_result_dir', type=str,
                        default='../output/sample/flow', help='optical flow result directory')
    parser.add_argument('--segment_result_dir', type=str,
                        default='../output/sample/segment', help='segmentation result directory')
    parser.add_argument('--output_dir', type=str,
                        default='../output/sample/final', help='output image directory')
    parser.add_argument('--loglevel', type=int, default=3,
                        help='log level:0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image')
    parser.add_argument('--resultimg_width', type=int,
                        default=1280, help='result image width.[pix]')
    parser.add_argument('--skip_frames', type=int,
                        default=0, help='skip frames at the beginning')
    args = parser.parse_args()

    moe = MovingObjectExtractor(args)
    moe.compute()
