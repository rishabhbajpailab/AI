import numpy as np
from dataclasses import dataclass, field

__all__ = ["NeuralNetwork", "SGD", "Adam", "TrainingResult"]


# ---------------------------------------------------------------------------
# Internal classes — implementation details, not part of the public API
# ---------------------------------------------------------------------------

class _Layer_Dense:
    # Forward pass computes the dot product of inputs and weights plus biases.
    # Backward pass computes gradients w.r.t. weights, biases, and inputs.
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class _Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class _Activation_Softmax:
    def forward(self, inputs):
        # Subtract row max before exponentiation to prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian, single_dvalues)


class _Loss:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))


class _Loss_CategoricalCrossEntropy(_Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = (-y_true / dvalues) / samples


class _SoftmaxCrossEntropyLoss:
    """Fused Softmax + Categorical Cross-Entropy for a numerically stable backward pass."""

    def __init__(self):
        self.activation = _Activation_Softmax()
        self.loss = _Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Describes the outcome of a :meth:`NeuralNetwork.train` call.

    Attributes
    ----------
    epochs : int
        Number of epochs the network was trained for.
    final_loss : float
        Cross-entropy loss on the training data at the final epoch.
    final_accuracy : float
        Classification accuracy on the training data at the final epoch,
        in the range [0.0, 1.0].
    loss_history : list[float]
        Loss value recorded at every epoch (length == epochs).
    accuracy_history : list[float]
        Accuracy value recorded at every epoch (length == epochs).
    """
    epochs: int
    final_loss: float
    final_accuracy: float
    loss_history: list = field(default_factory=list)
    accuracy_history: list = field(default_factory=list)


class SGD:
    """Stochastic Gradient Descent optimizer with optional momentum and learning-rate decay.

    Parameters
    ----------
    learning_rate : float
        Initial learning rate. Default is 1.0.
    decay : float
        Learning-rate decay applied each step (0 disables). Default is 0.0.
    momentum : float
        Momentum coefficient (0 disables). Default is 0.0.
    """

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def _pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def _update(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def _post_update(self):
        self.iterations += 1


class Adam:
    """Adam optimizer (Adaptive Moment Estimation).

    Parameters
    ----------
    learning_rate : float
        Initial learning rate. Default is 0.001.
    decay : float
        Learning-rate decay per step (0 disables). Default is 0.0.
    epsilon : float
        Small constant for numerical stability. Default is 1e-7.
    beta_1 : float
        Exponential decay rate for the first moment (mean). Default is 0.9.
    beta_2 : float
        Exponential decay rate for the second moment (variance). Default is 0.999.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def _pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def _update(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def _post_update(self):
        self.iterations += 1


def _validate_inputs(X, y=None):
    """Validate array inputs before training or inference.

    Parameters
    ----------
    X : array-like
        Input features. Must be a 2-D numeric numpy array.
    y : array-like or None
        Labels. When provided, must be a 1-D integer array or a 2-D
        one-hot float array, with the same number of rows as X.

    Raises
    ------
    TypeError
        If X or y are not numpy arrays.
    ValueError
        If X is not 2-D, X has a non-numeric dtype, y has an unsupported
        shape, or len(X) != len(y).
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array of shape (n_samples, n_features), got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"X must have a numeric dtype, got {X.dtype}")

    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")
        if y.ndim == 1:
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError(
                    f"1-D y must contain integer class labels, got dtype {y.dtype}. "
                    "Use y.astype(int) to convert."
                )
        elif y.ndim == 2:
            pass  # one-hot encoded; accepted as-is
        else:
            raise ValueError(f"y must be 1-D (class labels) or 2-D (one-hot), got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of samples; got {len(X)} and {len(y)}")


class NeuralNetwork:
    """A feedforward neural network for multi-class classification.

    Build the network layer by layer with :meth:`add_dense`, then call
    :meth:`train` to fit it to labelled data.  After training, use
    :meth:`predict` for inference and :meth:`accuracy` to evaluate performance.

    Examples
    --------
    >>> from nn import NeuralNetwork, Adam
    >>> net = NeuralNetwork()
    >>> net.add_dense(2, 128).add_dense(128, 3, activation="softmax")
    >>> result = net.train(X, y, Adam(learning_rate=0.02))
    >>> preds = net.predict(X)
    """

    def __init__(self) -> None:
        # List of (dense_layer, activation) pairs for all layers except the final one
        self._layers: list = []
        # Fused softmax + cross-entropy for the output layer (set on first train call)
        self._loss_activation: _SoftmaxCrossEntropyLoss | None = None
        self._trained: bool = False

    def add_dense(self, n_inputs: int, n_neurons: int, activation: str = "relu") -> "NeuralNetwork":
        """Append a fully-connected dense layer to the network.

        Parameters
        ----------
        n_inputs : int
            Number of input features for this layer.
        n_neurons : int
            Number of neurons (output units).
        activation : str
            Activation function: ``"relu"`` (default), ``"softmax"``, or
            ``"none"``.  The final layer must use ``"softmax"``.

        Returns
        -------
        NeuralNetwork
            Returns ``self`` to allow method chaining.

        Raises
        ------
        ValueError
            If ``activation`` is not one of the accepted values.
        """
        valid = {"relu", "softmax", "none"}
        if activation not in valid:
            raise ValueError(f"activation must be one of {valid}, got {activation!r}")

        layer = _Layer_Dense(n_inputs, n_neurons)
        if activation == "relu":
            act = _Activation_ReLU()
        elif activation == "softmax":
            act = _Activation_Softmax()
        else:
            act = None

        self._layers.append((layer, act, activation))
        return self

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer,
        epochs: int = 1000,
        print_every: int = 100,
    ) -> TrainingResult:
        """Train the network on labelled data.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape ``(n_samples, n_features)``.
        y : np.ndarray
            Class labels — either a 1-D integer array of shape ``(n_samples,)``
            or a 2-D one-hot encoded array of shape ``(n_samples, n_classes)``.
        optimizer : SGD or Adam
            Configured optimizer instance to use for weight updates.
        epochs : int
            Number of full passes over the training data. Default is 1000.
        print_every : int
            Print a progress line every this many epochs. Set to ``0`` to
            suppress all output. Default is 100.

        Returns
        -------
        TrainingResult
            Loss and accuracy recorded at every epoch, plus final values.

        Raises
        ------
        RuntimeError
            If no layers have been added before calling ``train()``.
        ValueError
            If the final layer does not use ``"softmax"`` activation, or if
            ``X`` / ``y`` have incompatible shapes or unsupported types.
        """
        _validate_inputs(X, y)

        if not self._layers:
            raise RuntimeError("No layers have been added. Call add_dense() before train().")

        _, _, last_activation = self._layers[-1]
        if last_activation != "softmax":
            raise ValueError(
                "The final layer must use activation='softmax' for categorical cross-entropy loss."
            )

        # Normalise labels to 1-D integer class indices once
        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        # Build the fused output loss/activation on the first training call
        if self._loss_activation is None:
            self._loss_activation = _SoftmaxCrossEntropyLoss()

        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            # --- Forward pass ---
            data = X
            for dense, act, _ in self._layers[:-1]:
                dense.forward(data)
                act.forward(dense.output)
                data = act.output

            last_dense, _, _ = self._layers[-1]
            last_dense.forward(data)
            loss = self._loss_activation.forward(last_dense.output, y)

            predictions = np.argmax(self._loss_activation.output, axis=1)
            accuracy = float(np.mean(predictions == y))

            loss_history.append(float(loss))
            accuracy_history.append(accuracy)

            if print_every > 0 and epoch % print_every == 0:
                print(
                    f"epoch: {epoch}, "
                    f"acc: {accuracy * 100:.3f}%, "
                    f"loss: {loss:.3f}, "
                    f"learning rate: {optimizer.current_learning_rate:.5f}"
                )

            # --- Backward pass ---
            self._loss_activation.backward(self._loss_activation.output, y)
            last_dense.backward(self._loss_activation.dinputs)
            upstream = last_dense.dinputs

            for dense, act, _ in reversed(self._layers[:-1]):
                act.backward(upstream)
                dense.backward(act.dinputs)
                upstream = dense.dinputs

            # --- Optimizer step ---
            optimizer._pre_update()
            for dense, _, _ in self._layers:
                optimizer._update(dense)
            optimizer._post_update()

        self._trained = True

        return TrainingResult(
            epochs=epochs,
            final_loss=loss_history[-1],
            final_accuracy=accuracy_history[-1],
            loss_history=loss_history,
            accuracy_history=accuracy_history,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass and return class index predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Integer array of predicted class indices, shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the network has not been trained yet.
        ValueError
            If ``X`` has an incompatible shape or unsupported dtype.
        """
        if not self._trained:
            raise RuntimeError("Network must be trained before calling predict(). Call train() first.")

        _validate_inputs(X)

        data = X
        for dense, act, _ in self._layers[:-1]:
            dense.forward(data)
            act.forward(dense.output)
            data = act.output

        last_dense, _, _ = self._layers[-1]
        last_dense.forward(data)
        # Use the activation directly — no y_true needed for inference
        self._loss_activation.activation.forward(last_dense.output)
        return np.argmax(self._loss_activation.activation.output, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy on the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape ``(n_samples, n_features)``.
        y : np.ndarray
            True labels, same format accepted by :meth:`train`.

        Returns
        -------
        float
            Fraction of correct predictions in ``[0.0, 1.0]``.
        """
        _validate_inputs(X, y)
        predictions = self.predict(X)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        return float(np.mean(predictions == y))


# ---------------------------------------------------------------------------
# Demo — run with: python nn.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import nnfs
    from nnfs.datasets import spiral_data

    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    net = NeuralNetwork()
    net.add_dense(2, 128, activation="relu")
    net.add_dense(128, 3, activation="softmax")

    result = net.train(X, y, optimizer=Adam(learning_rate=0.02, decay=1e-5), epochs=1001, print_every=100)
    print(f"\nTraining complete — final loss: {result.final_loss:.4f}, accuracy: {result.final_accuracy * 100:.2f}%")
