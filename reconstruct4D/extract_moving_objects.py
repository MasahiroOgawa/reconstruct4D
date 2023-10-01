from focus_of_expansion import CameraState
from focus_of_expansion import FoE
import opticalflow
import segmentator
import argparse
import cv2
import os
import sys
import numpy as np


def main(args):
    # preparation
    imgfiles = sorted([file for file in os.listdir(
        args.input_dir) if file.endswith('.jpg') or file.endswith('.png')])
    print(f"[INFO] reading input image files: {imgfiles}")
    optflow = opticalflow.UnimatchFlow(args.flow_result_dir)
    flow_analyzer = opticalflow.FlowAnalyzer(10*np.pi/180, args.loglevel)
    segm = segmentator.InternImageSegmentator(args.segment_result_dir)
    prev_img = None
    foe = FoE(loglevel=args.loglevel)
    result_imgw = 1280

    # process each image
    for img_name in imgfiles:
        img = cv2.imread(os.path.join(args.input_dir, img_name))

        if prev_img is None:
            prev_imgname = img_name
            prev_img = img
            continue

        if args.loglevel > 0:
            print(f"[INFO] processing {img_name} : {img.shape}")

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow, which is different from what we expect, which is t-1 to t flow.
        optflow.compute(prev_imgname)

        # currently jusr read regmentation result from corresponding image file name.
        segm.compute(img_name)

        # compute focus of expansion
        foe.compute(optflow.flow)

        # stopping erea is defined as foe.inlier_mask[row, col] = 0
        if foe.state == CameraState.STOPPING:
            foe.maxinlier_mask = foe.inlier_mask
        elif (foe.state == CameraState.STOPPING) or (foe.state == CameraState.ROTATING):
            flow_analyzer.compute(optflow.flow)
            foe.maxinlier_mask = flow_analyzer.flow_mask

        flow_mask_img = opticalflow.flow_mask_img(foe.maxinlier_mask)

        # overlay tranparently outlier_mask(moving object mask) into input image
        overlay_img = img.copy()//2
        if foe.state == CameraState.STOPPING:
            # increase the green channel for inlier_mask != 0 (moving)
            overlay_img[foe.maxinlier_mask != 0, 2] += 128
        else:
            # increase the red channel for outlier_mask == 2(outlier)
            overlay_img[foe.maxinlier_mask == 2, 2] += 128
        result_img = overlay_img

        # display the result
        if args.loglevel > 1:
            foe.draw(bg_img=optflow.flow_img)

            row1_img = cv2.hconcat([img, optflow.flow_img, segm.result_img])
            row2_img = cv2.hconcat([foe.result_img, flow_mask_img, result_img])
            result_img = cv2.vconcat([row1_img, row2_img])
            save_imgsize = (result_imgw, int(
                result_imgw*result_img.shape[0]/result_img.shape[1]))
            print(f"imgshape={img.shape}")
            print(f"save_imgsize={save_imgsize}")
            result_img = cv2.resize(
                result_img, save_imgsize)
            cv2.imshow('result', result_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # create the result image and save it.
        # change file extension to png
        save_imgname = img_name.replace('.jpg', '.png')
        cv2.imwrite(f"{args.output_dir}/{save_imgname}", result_img)

        # prepare for the next frame
        prev_imgname = img_name
        prev_img = img

    cv2.destroyAllWindows()


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
    args = parser.parse_args()

    main(args)
