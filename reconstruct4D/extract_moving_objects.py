import argparse
import cv2
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import reconstruct4D.opticalflow as opticalflow
from reconstruct4D.focus_of_expansion import FoE

def main(args):
    # preparation
    imgfiles = sorted([file for file in os.listdir(args.input_dir) if file.endswith('.jpg') or file.endswith('.png')])
    print(f"reading input image files: {imgfiles}")
    unimatch = opticalflow.UnimatchFlow(args.flow_result_dir)
    flow_analyzer = opticalflow.FlowAnalyzer()
    prev_img = None
    foe = FoE(f=3.45719e+03, loglevel = args.loglevel) # f is focal length of the camera. This value is for the sample images.

    # process each image
    for imgname in imgfiles:
        img = cv2.imread(os.path.join(args.input_dir, imgname))

        if prev_img is None:
            prev_imgname = imgname
            prev_img = img
            continue

        if args.loglevel > 0:
            print(f"{imgname} : {img.shape}")

        # currently just read flow from corresponding image file name.
        # unimatch flow at time t is t to t+1 flow, which is different from what we expect, which is t-1 to t flow.
        unimatch.compute(prev_imgname)
        
        # compute focus of expansion
        foe.compute(unimatch.flow, unimatch.flow_img)

        # treat the camera is rotating case
        if foe.is_camera_rotating:
            flow_analyzer.compute(unimatch.flow)
            foe.maxinlier_mask = flow_analyzer.flow_mask

        opticalflow.draw_flow_mask(foe.maxinlier_mask)

        # overlay tranparently outlier_mask into input image
        overlay_img = img.copy()//2
        # increase red channel for outlier_mask == 2
        overlay_img[foe.maxinlier_mask == 2, 2] += 128
        result_img = overlay_img

        # display the result
        if args.loglevel > 1:            
            result_img = cv2.vconcat([img, unimatch.flow_img])
            overlay_img = cv2.vconcat([foe.result_img, overlay_img])
            result_img = cv2.hconcat([result_img, overlay_img])
            result_img = cv2.resize(result_img, (1280,960))
            cv2.imshow('result', result_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


        # create result the image and save it.
        cv2.imwrite(f"{args.output_dir}/{imgname}", result_img)

        # prepare for the next frame
        prev_img = img

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract moving objects')
    parser.add_argument('--input_dir', type=str, default='../data/sample', help='input image directory')
    parser.add_argument('--flow_result_dir', type=str, default='../output/sample/flow', help='optical flow result directory')
    parser.add_argument('--output_dir', type=str, default='../output/sample', help='output image directory')
    parser.add_argument('--loglevel', type=int, default=3, help='log level:0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image')
    args = parser.parse_args()

    main(args)
