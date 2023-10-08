import os
import numpy as np
import cv2
import json


class Segmentator():
    def __init__(self, result_dir):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = result_dir
        self.classes = None
        self.load_classes()
        self.result_img = None  # segmentation image
        self.result_movingobj_img = None  # moving object image

    def compute(self, img_name):
        pass

    def draw(self, bg_img=None):
        self.bg_img = bg_img
        pass

    def load_classes(self):
        classes_file = os.path.join(
            self.this_dir, '..', 'data', 'classes.json')
        if os.path.exists(classes_file):
            self.classes = json.load(open(classes_file, 'r'))
        else:
            self.dump_classes_with_moving_prob()
            print(
                f"[ERROR] {classes_file} does not exist, so it is created newly in data/classes.json. \
                Please edit moving probability first.")
            exit()

    def dump_classes_with_moving_prob(self):
        """
        name: dump classes with moving probability.
        usage: Just use this function when you need class files when you change the class names.
        """
        self.class_names = json.load(
            open(os.path.join(self.result_dir, 'class_names.json'), 'r'))

        # create combined struct with id, class name and moving probagilities, which is 0 as default.
        self.classes = []
        for i, class_name in enumerate(self.class_names):
            self.classes.append(
                {'class_name': class_name, 'class_id': i, 'moving_prob': 0.0})

        # save classes with mobing probability
        self.classes_file = os.path.join(
            self.this_dir, '..', 'data', 'classes.json')
        json.dump(self.classes, open(self.classes_file, 'w'))


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

    def draw(self, bg_img=None):
        super().draw(bg_img)
        self.comp_movingobj_img()

    def comp_movingobj_img(self):
        self.result_movingobj_img = self.bg_img.copy()//2

        # get sky id
        sky_id = None
        for class_dict in self.classes:
            if class_dict['class_name'] == 'sky':
                sky_id = class_dict['class_id']
                break

        # draw sky mask as light blue in result_movingobj_img
        if sky_id is not None:
            sky_mask = (self.result_mask == sky_id)
            self.result_movingobj_img[sky_mask,
                                      0] += 128  # 0 means blue channel
