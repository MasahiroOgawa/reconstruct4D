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
    # display input images.
    print(imgname)
    print(img.shape)
    cv2.imshow('img', img)

    # currently just read flow from files.
    imgnum = imgname.split('.')[0]
    flow_file = os.path.join(flow.flow_dir, f"{imgnum}_pred.flo")
    print(f"flow_file={flow_file}")
    flow.compute(flow_file)

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    prev_img = img

cv2.destroyAllWindows()

# %%

# %%
if __name__ == '__main__':
    main()
