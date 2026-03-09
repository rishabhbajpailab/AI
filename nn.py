import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

#Class for layers, takes inputs of the number of inputs and the number of neurons
# has two methods and a constructor
# forward moves through the network in the forward direction 
# backward is for backpropagation, computing gradients w.r.t. weights, biases, and inputs
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp (inputs - np.max(inputs, axis=1, keepdims=True)) # exponentiation while subtracting from the largest in each row to protect from overflow 
        norm_values = exp_values/ np.sum(exp_values, axis=1, keepdims=True) # normalization
        self.output= norm_values
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
     # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
 def __init__(self, learning_rate=1., decay=0., momentum=0.):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum
 def pre_update_params(self):
     if self.decay:
         self.current_learning_rate = self.learning_rate * \
             (1. / (1. + self.decay *self.iterations))
 def update_params(self, layer):
     if self.momentum:
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        weight_updates = \
            self.momentum * layer.weight_momentums - \
            self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates
        # Bias updates
        bias_updates = \
            self.momentum * layer.bias_momentums - \
            self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates
     else:
        weight_updates = -self.current_learning_rate * \
                        layer.dweights
        bias_updates = -self.current_learning_rate * \
                        layer.dbiases
     layer.weights += weight_updates
     layer.biases += bias_updates
 def post_update_params(self):
     self.iterations += 1

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0, epsilon =1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations=0
        self.epsilon = epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *\
                (1. / (1. + self.decay * self.iterations))
    
    def update_params(self,layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
     self.iterations += 1


X,y = spiral_data(samples=100, classes=3)

optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

dense1 = Layer_Dense(2,128)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(128,3)
activation2 = Activation_Softmax()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


for epoch in range(1001):
    #forward pass

    #forward step on layer 1
    dense1.forward(X)
    #apply reduced linear activation to layer1
    activation1.forward(dense1.output)

    #forward step on layer 2
    dense2.forward(activation1.output)
    #apply SoftMax activation to Layer 2 (it is output layer)
    loss =loss_activation.forward(dense2.output,y)

    #generate predictions and calculate loss
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    
    #print output
    if not epoch%100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy*100:.3f}%, ' +
              f'loss: {loss:.3f}, ' +
              f'learning rate: {optimizer.current_learning_rate:.5f}')

    #back propogration

    #apply categorical cross entropy + softmax activation onto loss
    loss_activation.backward(loss_activation.output, y)
    #backwards step on layer 2
    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()