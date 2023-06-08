# %%
import sys
sys.path.append('..')
import reconstruct4D.opticalflow as opticalflow
import os
import cv2

def main():
    # preparation
    # TODO: use argparse. but currently to use jupyter, we cannot use argparse.
    image_dir = '/mnt/data/study/mine/computer_vision/todaiura/images/480p'

    imgfiles = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.png')])
    print(imgfiles)
    flow = opticalflow.UnimatchFlow()

    # process each image
    for imgname in imgfiles:
        img = cv2.imread(os.path.join(image_dir, imgname))

        # debug
        print(imgname)
        print(img.shape)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    flow.compute(None, None)

if __name__ == '__main__':
    main()

# %%
