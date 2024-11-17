import os
import numpy as np
import cv2
import json
from oneformersegmentator import OneFormerSegmentator
from PIL import Image


class Segmentator:
    def __init__(
        self,
        model_name=None,
        input_dir="../data/sample",
        result_dir="result",
        thre_static_prob=0.1,
        log_level=0,
    ):
        """
        model_name: segmentation model name. Currently only used for oneformer.
        input_dir: input image directory .
        result_dir: result directory where segmentation results are stored.
        thre_static_prob: threshold of moving probability to determine static objects.
        log_level: log level. 0: no log, 1: print log, 2: display image, 3: debug with detailed image, 4: debug with detailed image and stop every step.
        """
        # constants
        self.RESULT_DIR = result_dir
        self.INPUT_DIR = input_dir
        self.THRE_STATIC_PROB = thre_static_prob
        self.LOG_LEVEL = log_level
        self.THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        # variables
        self.model_name = model_name
        self.result_img = None
        self.result_mask = None
        self.moving_prob = None
        self.moving_prob_img = None
        self.result_movingobj_img = None
        self.classes = None
        self.load_classes()
        self.sky_id = None
        self._comp_sky_id()
        self.static_ids = []
        self._comp_static_ids()
        self.sky_mask = None
        self.nonsky_static_mask = None

    def compute(self, img_name):
        pass

    def _comp_moving_prob(self):
        self.moving_prob = np.zeros_like(self.result_mask, dtype=float)
        for row in range(self.result_mask.shape[0]):
            for col in range(self.result_mask.shape[1]):
                class_id = self.result_mask[row, col]
                self.moving_prob[row, col] = self.classes[class_id]["moving_prob"]

    def draw(self, bg_img=None):
        self.bg_img = bg_img
        self._draw_movingobj_img()
        self._draw_movingprob_img()

    def _draw_movingobj_img(self):
        self.result_movingmask_img = self.bg_img.copy() // 2

        # draw sky mask as light blue in result_movingobj_img
        self.result_movingmask_img[self.sky_mask, 0] += 128  # 0 means blue channel

        # draw moving object mask as blighter in result_movingobj_img
        moving_obj_mask = np.logical_not(
            np.logical_or(self.sky_mask, self.nonsky_static_mask)
        )
        self.result_movingmask_img[moving_obj_mask, :] += 128
        cv2.putText(
            self.result_movingmask_img,
            "moving mask",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    def _draw_movingprob_img(self):
        # draw moving probability as jet color in moving_prob_img.
        self.moving_prob_img = np.zeros(
            (self.moving_prob.shape[0], self.moving_prob.shape[1], 3), dtype=np.uint8
        )

        self.moving_prob_img = cv2.applyColorMap(
            (self.moving_prob * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.putText(
            self.moving_prob_img,
            "prior",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    def load_classes(self):
        classes_file = os.path.join(self.THIS_DIR, "..", "data", "classes.json")
        if os.path.exists(classes_file):
            self.classes = json.load(open(classes_file, "r"))
        else:
            self.dump_classes_with_moving_prob()
            print(
                f"[ERROR] {classes_file} does not exist, so it is created newly in data/classes.json. \
                Please edit moving probability first."
            )
            exit()

    def dump_classes_with_moving_prob(self):
        """
        name: dump classes with moving probability.
        usage: Just use this function when you need to change the class names in the class file.
        """
        self.class_names = json.load(
            open(os.path.join(self.RESULT_DIR, "class_names.json"), "r")
        )

        # create combined struct with id, class name and moving probagilities, which is 0 as default.
        self.classes = []
        for i, class_name in enumerate(self.class_names):
            self.classes.append(
                {"class_name": class_name, "class_id": i, "moving_prob": 0.0}
            )

        # save classes with mobing probability
        self.classes_file = os.path.join(self.THIS_DIR, "..", "data", "classes.json")
        json.dump(self.classes, open(self.classes_file, "w"))

    def _comp_sky_id(self):
        for class_dict in self.classes:
            if class_dict["class_name"] == "sky":
                self.sky_id = class_dict["class_id"]
                break

    def _comp_sky_mask(self):
        if self.sky_id is not None:
            self.sky_mask = self.result_mask == self.sky_id

    def _comp_static_ids(self):
        for class_dict in self.classes:
            if (class_dict["moving_prob"] < self.THRE_STATIC_PROB) and (
                class_dict["class_id"] != self.sky_id
            ):
                self.static_ids.append(class_dict["class_id"])

    def _comp_static_mask(self):
        if len(self.static_ids) > 0:
            self.nonsky_static_mask = np.zeros_like(self.result_mask, dtype=bool)
            for static_id in self.static_ids:
                self.nonsky_static_mask = np.logical_or(
                    self.nonsky_static_mask, (self.result_mask == static_id)
                )


class InternImageSegmentatorWrapper(Segmentator):
    # currenly just load the result from already processd direwhere input images are stored.ctory.
    def __init__(
        self, model_name=None, input_dir=None, result_dir="result", thre_static_prob=0.1, log_level=0
    ):
        """
        model_name: InternImageSegmentatorWrapper just load the result from already processed directory.
                    So this model_name is not used.
        """
        super().__init__(model_name, input_dir, result_dir, thre_static_prob, log_level)

    def compute(self, img_name):
        if self.LOG_LEVEL > 0:
            print(f"[INFO] InternImageSegmentator.compute({img_name})")

        # get segmentation image
        seg_imgfile = os.path.join(self.RESULT_DIR, img_name)
        self.result_img = cv2.imread(seg_imgfile)

        # get segmentation result
        imgnum = img_name.split(".")[0]
        seg_resultfile = os.path.join(self.RESULT_DIR, f"{imgnum}.npy")
        self.result_mask = np.load(seg_resultfile)

        self._comp_sky_mask()
        self._comp_static_mask()
        self._comp_moving_prob()


class OneFormerSegmentatorWrapper(Segmentator):
    def __init__(
        self,
        model_name="shi-labs/oneformer_coco_swin_large",
        input_dir="../data/sample",
        result_dir="result",
        thre_static_prob=0.1,
        log_level=0,
    ):
        super().__init__(model_name, input_dir, result_dir, thre_static_prob, log_level)
        self.task_type = "panoptic"

    def compute(self, img_name):
        if self.LOG_LEVEL > 0:
            print(f"[INFO] OneformerSegmentator.compute({img_name})")

        # load PIL image
        image = Image.open(os.path.join(self.INPUT_DIR, img_name))

        # run segmentation
        oneformer = OneFormerSegmentator(self.model_name, self.task_type)
        result_pilimg, segments_info = oneformer.inference(image)
        self.result_mask = np.array(result_pilimg)

        # convert PIL image to opencv image
        self.result_img = cv2.cvtColor(
            np.array(result_pilimg).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        self._comp_sky_mask()
        self._comp_static_mask()
        self._comp_moving_prob()
