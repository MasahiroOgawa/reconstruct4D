# ref: https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/oneformer
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch
import sys
import argparse


def run_segmentation(image, task_type="panoptic", model_name="shi-labs/oneformer_ade20k_dinat_large"):
    """Performs image segmentation based on the given task type.

    Args:
        image (PIL.Image): The input image.
        task_type (str): The type of segmentation to perform ('semantic', 'instance', or 'panoptic').

    Returns:
        PIL.Image: The segmented image.
        segments_info (dictionary): contains additional information on each segment.

    Raises:
        ValueError: If the task type is invalid.
    """
    processor = OneFormerProcessor.from_pretrained(
        model_name
    )  # Load once here
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_name
    )
    inputs = processor(images=image, task_inputs=[
        task_type], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    if task_type == "semantic":
        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        segments_info = None
    elif task_type == "instance":
        prediction = processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        predicted_map = prediction["segmentation"]
        segments_info = prediction["segments_info"]
    elif task_type == "panoptic":
        prediction = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        predicted_map = prediction["segmentation"]
        segments_info = prediction["segments_info"]
    else:
        raise ValueError(
            "Invalid task type. Choose from 'semantic', 'instance', or 'panoptic'"
        )

    # debug
    print(f"predicted_map = {predicted_map}")
    print(f"segments_info = {segments_info}")
    if segments_info is not None:
        for segment in segments_info:
            label = model.config.id2label[segment['label_id']]
            print(f"segment id = {segment['id']} : {label}")

    return predicted_map, segments_info


def show_image_comparison(image, predicted_map, segmentation_title):
    """Displays the original image and the segmented image side-by-side.

    Args:
        image (PIL.Image): The original image.
        predicted_map (PIL.Image): The segmented image.
        segmentation_title (str): The title for the segmented image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map)
    plt.title(segmentation_title + " Segmentation")
    plt.axis("off")
    plt.savefig("oneformer_segm.png")
    plt.show()


# run below sample if this file is called as main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image/image directory with oneFormer.")
    parser.add_argument("image_path", type=str, nargs="?", default=None, help="Path to the image to process. If not provided, a default image will be used")
    parser.add_argument("--model_name", type=str, default="shi-labs/oneformer_coco_swin_large", help="The model name to use for segmentation")
    # model_name = "shi-labs/oneformer_ade20k_swin_tiny"
    # model_name = "shi-labs/oneformer_coco_swin_large"
    # model_name = "shi-labs/oneformer_ade20k_swin_large"
    # model_name = "shi-labs/oneformer_coco_dinat_large"
    # model_name = "shi-labs/oneformer_ade20k_dinat_large"
    parser.add_argument("--segmentation_type", type=str, default="panoptic", help="The type of segmentation to perform ('semantic', 'instance', or 'panoptic')")
    args = parser.parse_args()
            
    if args.image_path is not None:
        image = args.image_path
    else:
        url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/resolve/main/ade20k.jpeg"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(response.raw)

    # run segmentation
    predicted_map, segments_info = run_segmentation(
        image, args.segmentation_type, args.model_name)
    show_image_comparison(image, predicted_map, args.segmentation_type)
