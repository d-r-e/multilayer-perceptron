import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from src.ft_math import mean, softmax, sigmoid, cross_entropy, delta_cross_entropy
from src.Perceptron import Dense, Relu, softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits

if __name__ == "__main__":
    df = pd.read_csv('./data/data.csv')
    split= 300
    X_train = normalize(np.array(df.iloc[:,2:]))
    X_test = X_train[split + 1:,:]
    y_train = np.squeeze(np.array(df['diagnosis'].map({'M':1,'B':0})))
    y_test = y_train[split+1:]

    network = []
    network.append(Dense(X_train.shape[1], 30))
    network.append(Relu())
    network.append(Dense(30, 10))
    network.append(Relu())
    network.append(Dense(10, 2))

    def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer. 
    
        activations = []
        input = X
        # Looping through each layer
        for l in network:
            activations.append(l.forward(input))
            # Updating input to last layer output
            input = activations[-1]
        
        assert len(activations) == len(network)
        return activations
        
    def predict(network,X):
        # Compute network predictions. Returning indices of largest Logit probability
        logits = forward(network,X)[-1]
        return logits.argmax(axis=-1)

    def train(network,X,y):
        # Train our network on a given batch of X and y.
        # We first need to run forward to get all layer activations.
        # Then we can run layer.backward going from last to first layer.
        # After we have called backward for all layers, all Dense layers have already made one gradient step.
        
        
        # Get the layer activations
        layer_activations = forward(network,X)
        layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
        logits = layer_activations[-1]
        
        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits,y)
        loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
        
        # Propagate gradients through the network
        # Reverse propogation as this is backprop
        for layer_index in range(len(network))[::-1]:
            layer = network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        return np.mean(loss)

    from tqdm import trange
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        if (len(inputs) != len(targets)):
            print(len(inputs))
            print(len(targets))
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.random.permutation(len(inputs))
        for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
    from IPython.display import clear_output
    train_log = []
    val_log = []
    for epoch in range(1000):
        for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=10,shuffle=True):
            train(network,x_batch,y_batch)
        
        train_log.append(np.mean(predict(network,X_train)==y_train))
        val_log.append(np.mean(predict(network,X_test)==y_test))
        
        clear_output()
        print("Epoch",epoch)
        print("Train accuracy:",train_log[-1])
        print("Val accuracy:",val_log[-1])
        plt.plot(train_log,label='train accuracy')
        plt.plot(val_log,label='val accuracy')
        plt.grid()
    plt.show()
    