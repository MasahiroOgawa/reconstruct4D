# ref: https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/oneformer

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import matplotlib.pyplot as plt
import torch
import numpy as np

class OneFormerSegmentator:
    """A class for segmenting images using OneFormer models.
    Args:
        model_name (str): The name of the model to use. You can select from the following models:
            - "shi-labs/oneformer_ade20k_swin_tiny"
            - "shi-labs/oneformer_coco_swin_large"
            - "shi-labs/oneformer_ade20k_swin_large"
            - "shi-labs/oneformer_coco_dinat_large"
            - "shi-labs/oneformer_ade20k_dinat_large"
        task_type (str): The type of segmentation task to perform. Choose from 'semantic', 'instance', or 'panoptic'.
    """
    def __init__(self, model_name="shi-labs/oneformer_coco_swin_large", task_type="panoptic"):
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.task_type = task_type

    def inference(self, image):
        self.image = image
        inputs = self.processor(images=self.image, task_inputs=[self.task_type], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.task_type == "semantic":
            self.predicted_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            self.segments_info = None
        elif self.task_type == "instance":
            prediction = self.processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            self.predicted_map = prediction["segmentation"]
            self.segments_info = prediction["segments_info"]
        elif self.task_type == "panoptic":
            prediction = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            self.predicted_map = prediction["segmentation"]
            self.segments_info = prediction["segments_info"]
        else:
            raise ValueError("Invalid task type. Choose from 'semantic', 'instance', or 'panoptic'")

        return self.predicted_map, self.segments_info

    def print_result(self):
        print(f"predicted_map = {self.predicted_map}")
        print(f"segments_info = {self.segments_info}")
        if self.segments_info is not None:
            for segment in self.segments_info:
                label = self.model.config.id2label[segment['label_id']]
                print(f"segment id = {segment['id']} : {label}")


    def print_alllabels(self):
        labels = self.model.config.id2label
        for id, name in labels.items():
            print(f"{id} : {name}")


    def show(self, with_label: bool = True):
        """Displays the original image and the segmented image side-by-side.

        Args:
            image (PIL.Image): The original image.
            predicted_map (PIL.Image): The segmented image.
            segmentation_title (str): The title for the segmented image.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(self.predicted_map)
        if with_label:
            self._draw_labels()
        plt.title(self.task_type + " Segmentation")
        plt.axis("off")
        plt.savefig("oneformer_segm.png")
        plt.show()
             
    def _draw_labels(self):
        for segment in self.segments_info:
            label = self.model.config.id2label[segment['label_id']]
            segment_id = segment['id']
            mask = self.predicted_map == segment_id  # Create a binary mask for the segment
            centroid_x, centroid_y = self._calculate_centroid(mask)
            # draw label on image
            plt.text(centroid_x, centroid_y, label, fontsize=12, color='black')  


    def _calculate_centroid(self, mask) -> tuple[float, float]:
        """Calculates the centroid of a binary mask.
        Args:
            mask (np.ndarray or torch.Tensor): The binary mask. 
        Returns:
            tuple[float, float]: The x, y coordinates of the centroid.
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        indices = np.argwhere(mask).astype(float)
        centroid = indices.mean(axis=0)
        return centroid[1], centroid[0]