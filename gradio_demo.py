"""
This script creates a Gradio GUI for detecting and classifying signature blocks in document images
using the SignatureBlockModel. It loads example images from the /assets directory, displays
bounding boxes in the result image, and shows signature crops with labels in a gallery.
"""

import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, List
import os

from scripts.signature_blocks import SignatureBlockModel

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def process_image(
    image: np.ndarray,
) -> Tuple[np.ndarray, str, List[Tuple[np.ndarray, str]]]:
    """
    Process an input image using the SignatureBlockModel.

    Args:
        image (np.ndarray): Input image as a numpy array.

    Returns:
        Tuple[np.ndarray, str, List[Tuple[np.ndarray, str]]]: Processed image, status, and list of crops with labels.
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)

    # Initialize the model
    model = SignatureBlockModel(pil_image)

    # Get processed image with boxes
    image_with_boxes = model.draw_boxes()

    # Get signature crops
    signature_crops = create_signature_crops(model)

    # Determine status
    labels = model.get_labels()
    if not labels.any():
        status = "Unsigned"
    elif all(label == 1 for label in labels):
        status = "Fully Executed"
    elif all(label == 2 for label in labels):
        status = "Unsigned"
    else:
        status = "Partially Executed"

    return np.array(image_with_boxes), status, signature_crops


def create_signature_crops(model: SignatureBlockModel) -> List[Tuple[np.ndarray, str]]:
    """
    Create individual images for each signature crop with label and score information.

    Args:
        model (SignatureBlockModel): The initialized SignatureBlockModel.

    Returns:
        List[Tuple[np.ndarray, str]]: List of tuples containing crop images and labels.
    """
    boxes = model.get_boxes()
    scores = model.get_scores()
    labels = model.get_labels()
    classes = model.classes

    crop_data = []

    for box, label, score in zip(boxes, labels, scores):
        crop = model.extract_box(box)
        # resized_crop = resize_crop(crop)
        label_text = f"{classes[label]}, Score: {score:.2f}"
        crop_data.append((crop, label_text))

    return crop_data


def resize_crop(crop: np.ndarray, max_height: int = 200) -> np.ndarray:
    """
    Resize a crop to a maximum height while maintaining aspect ratio.

    Args:
        crop (np.ndarray): Input crop as a numpy array.
        max_height (int): Maximum height for the crop.

    Returns:
        np.ndarray: Resized crop.
    """
    crop_image = Image.fromarray(crop)
    aspect_ratio = crop_image.width / crop_image.height
    new_height = min(crop_image.height, max_height)
    new_width = int(new_height * aspect_ratio)

    resized_crop = crop_image.resize((new_width, new_height), Image.LANCZOS)
    return np.array(resized_crop)


def load_examples():
    """
    Load example images from the /assets directory.

    Returns:
        List[List[str]]: List of example image paths.
    """
    examples = []
    for filename in os.listdir(ASSETS_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            examples.append([os.path.join(ASSETS_DIR, filename)])
    return examples


with gr.Blocks() as demo:
    gr.Markdown("# Signature Block Detection")
    gr.Markdown("Upload a document image to detect and classify signature blocks.")

    with gr.Row():
        input_image = gr.Image(label="Upload Document Image", type="numpy", height=500)
        output_image = gr.Image(label="Processed Image", type="numpy", height=500)

    signature_crops = gr.Gallery(
        label="Signature Crops",
        show_label=True,
        elem_id="gallery",
        columns=[6],
        rows=[1],
        allow_preview=True,
        object_fit="contain",
        height=250,
    )

    with gr.Row():
        status_box = gr.Textbox(label="Document Status")

    process_btn = gr.Button("Process Image")

    examples = gr.Examples(
        examples=load_examples(),
        inputs=input_image,
    )

    process_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, status_box, signature_crops],
    )

if __name__ == "__main__":
    demo.launch()
