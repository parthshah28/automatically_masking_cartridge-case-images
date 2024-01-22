# automatically_masking_cartridge-case-images

Perform image segmentation using the Spatially Adaptive Multi-Resolution Vision Transformer (SAM-ViT) model for forensic analysis of cartridge case images.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Model Information](#model-information)
- [Use Case](#use-case)
- [Analysis](#analysis)


## Introduction

The code performs image segmentation using the SAM-ViT model to identify specific features in cartridge case images. It utilizes a color-coded mask approach to highlight areas of interest.

## Features

- SAM-ViT model for image segmentation.
- Visualization of masks, bounding boxes, and IOU scores.
- Detailed information for each segmented region.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (specified in the code)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git

2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt

## Usage
To perform image segmentation and visualize the results, run the main script:

```bash
 python image_segmentation.py
```

## Code Overview
The code is structured with modular functions for showing masks, calculating IOU, and displaying masks on the input image. It uses the SAM-ViT model for mask generation.

## Model Information
The SAM-ViT model is a Spatially Adaptive Multi-Resolution Vision Transformer designed for image segmentation tasks. It is pre-trained on a large dataset and fine-tuned for forensic analysis.

## Use Case
This project is specifically designed for forensic analysis of cartridge case images. It assists forensic analysts in identifying and analyzing critical features such as breech-face impressions, aperture shears, firing pin impressions, and firing pin drags.

## Analysis
The code provides a detailed analysis of each segmented region, including area, bounding box coordinates, and predicted IOU scores. Analysis Report()
