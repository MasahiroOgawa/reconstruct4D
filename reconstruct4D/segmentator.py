import os
import numpy as np
import cv2


class Segmentator():
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def compute():
        pass

    def show():
        pass


class InternImageSegmentator(Segmentator):
    # currenly just load the result from already processd directory.
    def __init__(self, result_dir):
        super().__init__(result_dir)

    def compute(self, img_name):
        imgnum = img_name.split('.')[0]
        # get segmentation image
        seg_imgfile = os.path.join(self.result_dir, f"{imgnum}.jpg")
        self.result_img = cv2.imread(seg_imgfile)

        # get segmentation result
        seg_resultfile = os.path.join(self.result_dir, f"{imgnum}.npy")
        self.result_mask = np.load(seg_resultfile)
