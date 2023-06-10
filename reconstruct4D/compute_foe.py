import os
import cv2
import sys
sys.path.append('..')
import reconstruct4D.opticalflow as opticalflow
import reconstruct4D.focus_of_expansion as focus_of_expansion

def main():
    # paramaters
    # TODO: use argparse. but currently to use jupyter, we cannot use argparse.
    image_dir = '/mnt/data/study/mine/computer_vision/todaiura/images/480p'

    # preparation
    imgfiles = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.png')])
    print(f"reading input image files: {imgfiles}")
    unimatch = opticalflow.UnimatchFlow()
    prev_img = None
    foe = focus_of_expansion.FoE(f=1000)

    # process each image
    for imgname in imgfiles:
        img = cv2.imread(os.path.join(image_dir, imgname))

        if prev_img is None:
            prev_img = img
            continue

        # debug
        print(f"{imgname} : {img.shape}")

        # currently just read flow from corresponding image file name.
        unimatch.compute(imgname)
        
        # display the esult
        result_img = cv2.vconcat([img, unimatch.flow_img])
        cv2.imshow('result', result_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        # compute focus of expansion
        foe.compute(unimatch.flow, unimatch.flow_img)

        # prepare for the next frame
        prev_img = img

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

