import os
import numpy as np
import cv2


class Segmentator():
    def __init__(self, config_file=None, checkpoint_file=None, device='cuda:0', color_palette='ade20k', opacity=0.5):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.color_palette = color_palette
        self.opacity = opacity
        self.model = None

    def compute():
        pass

    def show():
        pass


class InternImageSegmentator(Segmentator):
    # currenly just load the result from already processd directory.
    def __init__(self, result_dir, palette='ade20k', opacity=0.5):
        self.result_dir = result_dir
        self.palette = palette
        self.opacity = opacity

    def compute(self, img_name):
        imgnum = img_name.split('.')[0]
        # get segmentation image
        seg_imgfile = os.path.join(self.result_dir, f"{imgnum}.jpg")
        self.result_img = cv2.imread(seg_imgfile)

        # get segmentation result
        seg_resultfile = os.path.join(self.result_dir, f"{imgnum}.npy")
        self.result_mask = np.load(seg_resultfile)
