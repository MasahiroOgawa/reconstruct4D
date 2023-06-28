import os
import cv2
import sys
sys.path.append('..')
import reconstruct4D.opticalflow as opticalflow
from reconstruct4D.focus_of_expansion import FoE

def main():
    # paramaters
    # TODO: use argparse. but currently to use jupyter, we cannot use argparse.
    image_dir = '/mnt/data/study/mine/computer_vision/todaiura/images/480p'
    loglevel = 3 # 0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image

    # preparation
    imgfiles = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.png')])
    print(f"reading input image files: {imgfiles}")
    unimatch = opticalflow.UnimatchFlow()
    flow_analyzer = opticalflow.FlowAnalyzer()
    prev_img = None
    foe = FoE(f=3.45719e+03, loglevel = loglevel)

    # process each image
    for imgname in imgfiles:
        img = cv2.imread(os.path.join(image_dir, imgname))

        if prev_img is None:
            prev_img = img
            continue

        if loglevel > 0:
            print(f"{imgname} : {img.shape}")

        # currently just read flow from corresponding image file name.
        unimatch.compute(imgname)
        
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
        if loglevel > 1:            
            result_img = cv2.vconcat([img, unimatch.flow_img])
            overlay_img = cv2.vconcat([foe.result_img, overlay_img])
            result_img = cv2.hconcat([result_img, overlay_img])
            result_img = cv2.resize(result_img, (640,480))
            cv2.imshow('result', result_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


        # create result the image and save it.
        cv2.imwrite(f"../output/{imgname}", result_img)

        # prepare for the next frame
        prev_img = img

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

