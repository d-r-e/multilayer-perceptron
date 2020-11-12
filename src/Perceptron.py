from src.ft_math import sigmoid, relu, softmax
from src.ft_math import cross_entropy as cost
import numpy as np
import sklearn.preprocessing


# https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

class Layer:

    # A building block. Each layer is capable of performing two things:
    # - Process input to get output:
    """ output = layer.forward(input) """
    # - Propagate gradients through itself:
    """ grad_input = layer.backward(input, grad_output) """
    # Some layers also have learnable parameters which they update during
    # layer.backward.

    def __init__(self):
        # Here we can initialize layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass

    def forward(self, input):
        # Takes input data of shape [batch, input_units],
        # returns output data [batch, output_units]
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        # Performs a backpropagation step through the layer, with respect to
        # the given input.
        # To compute loss gradients w.r.t input, we need to apply chain
        # rule (backprop):
        """ d loss / d x  = (d loss / d layer) * (d layer / d x) """
        # Luckily, we already receive d loss / d layer as input, so you only
        # need to multiply it by d layer / d x.
        # If our layer has parameters (e.g. dense layer), we need to update
        # them here using d loss / d layer
        # The gradient of a dummy layer is just grad_output, but we'll write
        # more explicitly
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)  # chain rule


class Relu(Layer):

    def __init__(self):
        pass

    def forward(self, input):
        relu_forward = np.maximum(0,input)
        return relu_forward

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad


class Dense(Layer):

    def __init__(self, input_units, output_units, learning_rate=0.1):
        """ A dense layer performs a learned affine transformation:
            f(x) = <W*x> + b """
        self.learning_rate = learning_rate
        self.weights = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2/(input_units + output_units)),
            size=(input_units, output_units)
            )
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """ Perform an affine transformation:
            f(x) = <W*x> + b """

        # input shape: [batch, input_units]
        # output shape: [batch, output units]

        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        assert grad_weights.shape == self.weights.shape
        assert grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return ( -ones_for_answers + softmax) / logits.shape[0]


if __name__ == "__main__":
    """ n = 8
    l0 = Dense(n, 2)
    X = np.eye(n)
    print(l0.forward(X))
    print(l0.backward(X, l0.forward(X)))
    print() """
    x2 = np.array([1, 1, 1, 1, 1, 1])
    x3 = x2
    print(scross(x2, x3))
