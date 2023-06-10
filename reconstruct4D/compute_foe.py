# %%
import os
import cv2
import sys
sys.path.append('..')
import reconstruct4D.opticalflow as opticalflow

# def main():

# paramaters
# TODO: use argparse. but currently to use jupyter, we cannot use argparse.
image_dir = '/mnt/data/study/mine/computer_vision/todaiura/images/480p'

# preparation
imgfiles = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg') or file.endswith('.png')])
print(imgfiles)
flow = opticalflow.UnimatchFlow()
prev_img = None

# %%imgname
# process each image
for imgname in imgfiles:
    img = cv2.imread(os.path.join(image_dir, imgname))

    if prev_img is None:
        prev_img = img
        continue

    # debug
    print(img.shape)

    # currently just read flow from corresponding image file name.
    flow.compute(imgname)
    
    # display result
    result_img = cv2.vconcat([img, flow.flow_img])
    cv2.imshow('result', result_img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    prev_img = img

cv2.destroyAllWindows()

# %%

# %%
if __name__ == '__main__':
    main()
