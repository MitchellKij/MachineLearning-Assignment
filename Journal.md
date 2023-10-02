# Learning Journal 

## Introduction to project
The purpose of this project/ journal is to be a referencing tool as part of the live demonstration for the assessment. It will contain a more detailed overview of the work done. 

The project is implimentation of a neural network from scratch. To do this the main steps of the project are
* Extraction of dataset
* Pre-processing
* The neural network implimentation 
* Evaluation of results. 

The data that the network is being tested on is the MNIST dataset, which is a commonly used machine learning dataset of handwritten digits. 


## Implimentation

Flattening the Images: Each image in the MNIST dataset is a 2D array (28x28 pixels). For simplicity, 2D array is flattened into a 1D array of 784 numbers. 

$\text{Flattened Image} \in \mathbb{R}^{784}$

The pixel values for each image in the dataset are grayscale values ranging from 0 to 255. We normalize these values to the range [0, 1] by dividing each pixel value by 255. This helps in faster and more stable convergence during the training process.

$\text{Normalized Pixel Value} = \frac{\text{Original Pixel Value}}{255}$


## Forward Propagation

Forward propagation serves as the neural network's inference phase where the model takes in inputs and passes them through each layer to produce an output. It is during this phase that the neural network makes its initial guess about what the output should be.

Linear Transformation: This operation is represented mathematically as $Z = XW + b$, where $X$ is the input, $W$ is the weight matrix, and $b$ is the bias vector. This step transforms the input into a form that can be activated by the subsequent activation function.

In the code this is done in the `forward_layer()` function 
```python 
Z = np.dot(A_prev, W) + b
```


**Activation Functions:** After the linear transformation, the resulting value is passed through an activation function. We used ReLU (Rectified Linear Unit) is used for the hidden layer and Softmax for the output layer.

$\text{ReLU}(x) = \max(0, x) $

$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $

Output: The final layer's activation output is the network's prediction, which is then compared to the actual label to compute the loss.

## Backward Propagation


Backward propagation is where the neural network learns from its errors by adjusting its weights and biases to minimize the loss function. This phase is crucial for the optimization of the neural network.

**Compute Gradients**

The first step in backpropagation is to calculate the gradients of the loss function with respect to each weight and bias. This is performed through the chain rule of calculus. In the code, these gradients are calculated in the backward_propagation() function and stored in variables dW1, db1, dW2, and db2.

**Update Parameters**

The gradients are then used to update the weights and biases in the network. The code employs Mini-batch Gradient Descent for this, implemented in the update_parameters() function.

**Error Propagation**

The term 'backward' in backpropagation refers to the computation of gradients starting from the output layer and moving backward through the network. This allows for the efficient computation of gradients for each layer based on the gradients of the layer that follows it.
Normalisation

Gradients are normalized by the number of samples in the batch to ensure that the weight and bias updates are independent of the batch size. This is explicitly done in the `backward_propagation()` function in the code.

**Learning Rate (α)**

The learning rate is a hyperparameter that controls the size of the updates to the weights and biases. In the code, this is set as learning_rate and is used in the update_parameters() function to perform the updates.

**Significance of Gradients**

Gradients signify the partial derivatives of the loss function with respect to each parameter. They indicate the direction and magnitude by which each parameter should be updated to minimize the loss. In the code, they are computed in `backward_propagation()` and applied in `update_parameters()`.

**Parameter Update Strategy**

In line with the principles of gradient descent, the actual parameters `(W1, b1, W2, b2)` are updated in the direction opposite to the gradient. This is based on the mathematical premise that a function decreases most rapidly if one moves in the direction of the negative gradient.

## Loss
A loss function is pivotal in training neural networks, as it quantifies the disparity between the predicted and actual outcomes. In machine learning models, particularly neural networks, the loss function serves as a key performance metric, providing a numerical value that the model aims to minimize during training.

In the context of the code, the Categorical Cross-Entropy Loss is employed for a multi-class classification problem. 

The central goal of training a neural network is to adjust its parameters so that the model approximates the true, underlying relationship between the inputs and outputs as closely as possible. This is achieved by minimizing the loss function. A lower loss value indicates that the model's predictions are more aligned with the actual data, making the model more reliable and effective in making future predictions.

## Layers
In a neural network, layers are structured units of nodes or neurons that transform the input data. Each layer performs specific operations dictated by activation functions and modifiable parameters (weights and biases). The network architecture in the code comprises:

* Input Layer: This is the entry point of the network where each neuron corresponds to one feature of the dataset. In the code, the input layer implicitly has 784 neurons, matching the number of features in the MNIST dataset.
* Hidden Layer: Hidden layers reside between the input and output layers, performing transformations on the input data. In this network, there is one hidden layer with 128 neurons, characterized by ReLU (Rectified Linear Unit) activation. The variables W1 and b1 hold the weights and biases for this layer, respectively.
* Output Layer: This is the final layer of the network, and it typically transforms the values from the last hidden layer into output values that make sense for the given problem. In this case, the output layer has 10 neurons, each representing a class of digits (0-9). The Softmax function is applied to convert these values into probabilities. The weights and biases for this layer are stored in W2 and b2, respectively.

Each layer in the neural network serves to gradually transform the raw input into a form that makes it easier to produce the desired output. The transformations are governed by the layer's activation functions and its learnable parameters.


## Optimisation 
The optimisation algorithm used in the code is a basic form of Gradient Descent, specifically Mini-batch Gradient Descent. This is evident from the loop structure within the train_neural_network() function, where the model parameters (weights and biases) are updated incrementally for each mini-batch of data.

Mini-batch Gradient Descent strikes a balance between these two approaches. It divides the dataset into smaller batches and updates the model parameters for each batch

In the `train_neural_networks()` function 
```python
for i in range(0, len(X_train), batch_size):
    # Mini-batch data
    X_batch = X_train_shuffled[i:i + batch_size]
    y_batch = y_train_shuffled[i:i + batch_size]
```

Forward and Backward Propagation: For each batch, the network undergoes a forward and a backward propagation to compute the gradients.
```python 
# Forward propagation
A1, Z1, A2, Z2 = forward_propagation(X_batch, W1, b1, W2, b2)
# Backward propagation
dW1, db1, dW2, db2 = backward_propagation(A1, Z1, A2, Z2, X_batch, y_batch, W2)
```

Parameter Update: After obtaining the gradients, the parameters are updated.
```python 
# Update parameters
update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
```
The learning_rate controls the step size during the optimisation process. It's set prior to the training loop and is used in the update_parameters() function to adjust the weights and biases.

#### References/ helpful links:
[ChatGPT session](https://chat.openai.com/share/0b8168ea-5d2a-497d-967e-c129e2424fcf) 

https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch 