import torch
import numpy as np
import cv2
from argparse import ArgumentParser


def create_heatmap(activations, channel=None, width=224, height=224):
    if channel is None:  # Get average over all channels
        heatmap = torch.mean(activations.squeeze(), dim=0)
    else:
        heatmap = activations.squeeze()[channel]

    heatmap = np.maximum(0, heatmap)  # Applying ReLU
    if torch.max(heatmap) != 0:
        heatmap /= torch.max(heatmap)  # Normalizing to be in range of [0,1]

    heatmap = np.asarray(heatmap)
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def get_image_heatmap_combination(img, heatmap, heatmap_ratio=0.45, **kwargs):
    combined_img = np.uint8(heatmap_ratio * heatmap + (1 - heatmap_ratio) * img)

    channel = kwargs.get("channel", None)
    if channel is not None:
        channel = kwargs["channel"]
        cv2.putText(
            combined_img,
            f"Feature map #{channel:03d}",
            (int(combined_img.shape[1] * 0.25), int(combined_img.shape[0] * 0.97)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 0),
            2,
            3,
        )

    return combined_img


def get_weighted_activations(activations, gradients):
    """
    for VGG-19 the shape of tensors are as follows:
        activations.shape --> 512 x 14 x 14
        gradients.shape --> 1 x 512 x 14 x 14
    """
    pooled_gradients = torch.mean(gradients.squeeze(), dim=[1, 2])
    for ch in range(activations.shape[0]):
        activations[ch, :, :] *= pooled_gradients[ch]

    return activations


def initialize_parser():
    parser = ArgumentParser(description="Grad-CAM implementation in PyTorch")

    parser.add_argument(
        "inference-image-path",
        type=str,
        help="path to the image you intend to inspect",
    )

    parser.add_argument(
        "--classes-index-path",
        type=str,
        default="./imagenet-classes-index-path.json",
        help="path to the JSON file providing the labels/classes for each class-ID in ImageNet",
    )

    parser.add_argument(
        "--heatmap-ratio",
        type=float,
        default=0.45,
        help="the ratio of heatmap (in range of [0,1]) used while combining original image with the heatmap --> combined_img = RATIO*heatmap + (1-RATIO)*orig_img",
    )

    parser.add_argument(
        "--save-output",
        action="store_true",
        help="whether to save the GIF produced by appending different activation channels",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="output_results",
        help="the filename of the output",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="whether to print the workflow progress",
        action="store_true",
    )

    return parser
