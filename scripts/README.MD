<div align="left">
<h1>Signature-Detection</h1>
</div>
<a href='https://huggingface.co/spaces/jordandavis/Signature-Detection'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>
<br>
<br>



A PyTorch-based model for detecting and classifying signature blocks in documents, particularly useful for determining the execution status of agreements and contracts.

## Overview

This repository contains a machine learning model trained to detect signature boxes in documents and classify them as either signed or unsigned. The primary use case is to automatically classify agreements and contracts as unsigned, partially executed, or fully executed at scale.

## Features

- Detect multiple signature blocks on a single page
- Classify signature blocks as signed or unsigned
- Determine the execution status of multi-page documents
- Process multiple documents in batch for scalable classification

## Installation

To use this project, you'll need Python 3.7+ and PyTorch. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/MnLgt/signature-detection.git
cd signature-detection
pip install -r requirements.txt
```

## Usage

The main components of this project are:

1. `SignatureBlockModel`: The core model for detecting and classifying signature blocks.
2. `execution_status`: A function to process documents and determine their execution status.

### Basic Usage

```python
from scripts.signature_blocks import SignatureBlockModel
from scripts.execution_status import execution_status

# Process a single image
model = SignatureBlockModel("path/to/your/image.jpg")
boxes = model.get_boxes()
labels = model.get_labels_names()

# Process multiple pages of a document
images = ["path/to/page1.jpg", "path/to/page2.jpg", "path/to/page3.jpg"]
num_sig_pages, status, boxes, crops = execution_status(images)
print(f"Number of signature pages: {num_sig_pages}")
print(f"Execution status: {status}")
```

### Advanced Usage

For more advanced usage, including visualizing detected boxes and processing PDFs, refer to the example in `SignatureDetection.ipynb`.

## Model Details

The signature detection model is based on a Faster R-CNN architecture with a MobileNet V3 Large FPN backbone. It has been trained to detect three classes:

- NOTHING (0)
- SIGNED_BLOCK (1)
- UNSIGNED_BLOCK (2)

The model uses non-maximum suppression (NMS) to filter overlapping detections and a score threshold to ensure high-quality predictions.