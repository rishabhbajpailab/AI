# AI Learning Project

A collection of ML/AI implementations built from scratch for learning purposes.

## Contents

- **nn.py** — Neural network built from scratch using NumPy. Includes dense layers, ReLU/Softmax activations, categorical cross-entropy loss, and SGD/Adam optimizers. Trains on a spiral classification dataset.
- **cv.py** — OpenCV template matching demo. Finds a template image within a source image using multiple matching methods (selectable via trackbar).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Neural Network Training

```bash
python nn.py
```

Trains a two-layer network (2 → 128 → 3) on a spiral dataset for 1000 epochs, printing loss and accuracy every 100 steps.

### OpenCV Template Matching

```bash
python cv.py <source_image> <template_image> [<mask_image>]
```

Opens an interactive window with a trackbar to cycle through the available matching methods (TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED).

## Dependencies

See `requirements.txt`. Key packages:

- `numpy` — array math for the neural network
- `nnfs` — helper utilities (spiral dataset generator)
- `opencv-python` — image processing for cv.py
