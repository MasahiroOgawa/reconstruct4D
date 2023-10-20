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
        self.result_imgw = args.result_imgw
        # variables
        self.imgfiles = sorted([file for file in os.listdir(
            args.input_dir) if file.endswith('.jpg') or file.endswith('.png')])
        print(f"[INFO] reading input image files: {self.imgfiles}")
        self.optflow = opticalflow.UnimatchFlow(args.flow_result_dir)
        self.undominantflow = opticalflow.UndominantFlowAngleExtractor(
            10*np.pi/180, args.loglevel)
        self.segm = segmentator.InternImageSegmentator(args.segment_result_dir)
        self.foe = FoE(loglevel=args.loglevel)
        self.prev_imgname = None
        self.prev_img = None
        self.cur_imgname = None
        self.cur_img = None

    def compute(self):
        # process each image
        for self.cur_imgname in self.imgfiles:
            self.process_image()
            self.draw()

        cv2.destroyAllWindows()

    def process_image(self):
        self.cur_img = cv2.imread(
            os.path.join(args.input_dir, self.cur_imgname))

        if self.prev_img is None:
            self.prev_imgname = self.cur_imgname
            self.prev_img = self.cur_img
            return

        if args.loglevel > 0:
            print(
                f"[INFO] processing {self.cur_imgname} : {self.cur_img.shape}")

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow, which is different from what we expect, which is t-1 to t flow.
        self.optflow.compute(self.prev_imgname)

        # currently jusr read regmentation result from corresponding image file name.
        self.segm.compute(self.cur_imgname)

        # compute focus of expansion
        self.foe.compute(self.optflow.flow,
                         self.segm.sky_mask, self.segm.nonsky_static_mask)

        # stopping erea is defined as foe.inlier_mask[row, col] = 0
        if self.foe.state == CameraState.ROTATING:
            self.undominantflow.compute(
                self.optflow.flow, self.segm.nonsky_static_mask)
            self.foe.maxinlier_mask = self.undominantflow.flow_mask

    def draw(self) -> None:
        if self.foe.maxinlier_mask is None:
            return
        flow_mask_img = opticalflow.flow_mask_img(self.foe.maxinlier_mask)

        # overlay tranparently outlier_mask(moving object mask) into input image
        overlay_img = self.cur_img.copy()//2
        if self.foe.state == CameraState.STOPPING:
            # increase the red channel for inlier_mask != 0 (moving)
            overlay_img[self.foe.maxinlier_mask != 0, 2] += 128
        else:
            # increase the red channel for outlier_mask == 2(outlier)
            overlay_img[self.foe.maxinlier_mask == 2, 2] += 128
        result_img = overlay_img

        if args.loglevel > 1:
            # display the result
            self.foe.draw(bg_img=self.optflow.flow_img)
            self.segm.draw(bg_img=self.cur_img)

            row1_img = cv2.hconcat(
                [self.cur_img, self.optflow.flow_img, self.segm.seg_img])
            row2_img = cv2.hconcat(
                [self.segm.result_movingobj_img, self.foe.result_img, flow_mask_img])
            row3_img = cv2.hconcat(
                [self.foe.result_img, flow_mask_img, result_img])
            result_img = cv2.vconcat([row1_img, row2_img, row3_img])
            # resize keeping combined image aspect ratio
            save_imgsize = (self.result_imgw, int(
                self.result_imgw*result_img.shape[0]/result_img.shape[1]))
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

        # prepare for the next frame
        self.prev_imgname = self.cur_imgname
        self.prev_img = self.cur_img


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
    parser.add_argument('--result_imgw', type=int,
                        default=1280, help='result image width.[pix]')
    args = parser.parse_args()

    moe = MovingObjectExtractor(args)
    moe.compute()
