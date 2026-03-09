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

Run the built-in demo directly:

```bash
python nn.py
```

Trains a two-layer network (2 → 128 → 3) on a spiral dataset for 1001 epochs, printing loss and accuracy every 100 steps, then prints a `TrainingResult` summary.

Or import `nn` as a library in your own script:

```python
import numpy as np
from nn import NeuralNetwork, Adam, SGD, TrainingResult

# Build the network
net = NeuralNetwork()
net.add_dense(2, 128, activation="relu")   # hidden layer
net.add_dense(128, 3, activation="softmax") # output layer (must be softmax)

# Train — returns a TrainingResult dataclass
result: TrainingResult = net.train(X, y, optimizer=Adam(learning_rate=0.02, decay=1e-5), epochs=1000)

print(f"Final loss:     {result.final_loss:.4f}")
print(f"Final accuracy: {result.final_accuracy * 100:.2f}%")
print(f"Loss history:   {result.loss_history[:5]} ...")  # one value per epoch

# Inference — returns a 1-D int array of class indices, shape (n_samples,)
predictions = net.predict(X)

# Convenience accuracy helper
acc = net.accuracy(X, y)
```

**Public API:**

| Symbol | Type | Description |
|---|---|---|
| `NeuralNetwork` | class | Build, train, and run inference |
| `NeuralNetwork.add_dense(n_inputs, n_neurons, activation)` | method | Add a layer; `activation` ∈ `{"relu", "softmax", "none"}` |
| `NeuralNetwork.train(X, y, optimizer, epochs, print_every)` | method | Train the network; returns `TrainingResult` |
| `NeuralNetwork.predict(X)` | method | Returns int array of class predictions |
| `NeuralNetwork.accuracy(X, y)` | method | Returns float accuracy in [0, 1] |
| `Adam` | class | Adam optimizer (`learning_rate`, `decay`, `epsilon`, `beta_1`, `beta_2`) |
| `SGD` | class | SGD optimizer (`learning_rate`, `decay`, `momentum`) |
| `TrainingResult` | dataclass | Output of `train()`: `epochs`, `final_loss`, `final_accuracy`, `loss_history`, `accuracy_history` |

**Input / output contracts:**

| Parameter | Expected type | Shape |
|---|---|---|
| `X` | `np.ndarray` (numeric dtype) | `(n_samples, n_features)` |
| `y` | `np.ndarray` — integer labels or one-hot floats | `(n_samples,)` or `(n_samples, n_classes)` |
| `predictions` | `np.ndarray` (int) | `(n_samples,)` |
| `accuracy` | `float` | scalar in `[0.0, 1.0]` |

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
