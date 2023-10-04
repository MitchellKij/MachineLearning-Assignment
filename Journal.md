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
```python 
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
```

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

## Data dictionary 
### Functions 
Functions

* load_mnist_images(filename):
    Input Arguments:
        filename (String): The path to the MNIST images file.
    Output: NumPy Array containing the images.
    Description: Reads MNIST image data from a given file and returns it as a NumPy array.

* load_mnist_labels(filename):
    Input Arguments:
        filename (String): The path to the MNIST labels file.
    Output: NumPy Array containing the labels.
    Description: Reads MNIST label data from a given file and returns it as a NumPy array.

* relu(Z):
    Input Arguments:
        Z (NumPy Array): Pre-activated values.
    Output: NumPy Array after applying ReLU.
    Description: Applies the ReLU activation function to a given array.

* softmax(Z):
    Input Arguments:
        Z (NumPy Array): Pre-activated values.
    Output: NumPy Array after applying softmax.
    Description: Applies the softmax activation function to a given array.

* forward_layer(A_prev, W, b, activation):
    Input Arguments:
        A_prev (NumPy Array): Activated values from the previous layer.
        W (NumPy Array): Weights of the current layer.
        b (NumPy Array): Biases of the current layer.
        activation (String): Type of activation function ('relu' or 'softmax').
    Output: Activated values and pre-activated values for the current layer.
    Description: Performs forward propagation for a single layer.

* forward_propagation(X, W1, b1, W2, b2):
    Input Arguments:
        X (NumPy Array): Input data.
        W1, W2 (NumPy Arrays): Weights for hidden and output layers.
        b1, b2 (NumPy Arrays): Biases for hidden and output layers.
    Output: Intermediate variables used for forward propagation.
    Description: Performs forward propagation through the entire network.

* one_hot_encode(labels, num_classes):
    Input Arguments:
        labels (NumPy Array): Array of labels to encode.
        num_classes (Integer): Total number of unique classes.
    Output: One-hot encoded labels.
    Description: One-hot encodes the given labels.

* backward_layer(dA, Z, A_prev, W, activation):
    Input Arguments:
        dA (NumPy Array): Gradient of the loss with respect to post-activation values.
        Z (NumPy Array): Pre-activated values for the layer.
        A_prev (NumPy Array): Activated values from the previous layer.
        W (NumPy Array): Weights of the current layer.
        activation (String): Type of activation function ('relu' or 'softmax').
    Output: Gradients for the previous layer, current weights, and biases.
    Description: Performs backward propagation for a single layer.

* backward_propagation(A1, Z1, A2, Z2, X, y, W2):
    Input Arguments:
        A1, Z1, A2, Z2 (NumPy Arrays): Intermediate variables from forward propagation.
        X (NumPy Array): Input data.
        y (NumPy Array): True labels.
        W2 (NumPy Array): Weights for the output layer.
    Output: Gradients for weights and biases.
    Description: Performs backward propagation through the entire network.

* cross_entropy_loss(A, y):
    Input Arguments:
        A (NumPy Array): Predictions from the model.
        y (NumPy Array): True labels.
    Output: Loss value.
    Description: Computes the categorical cross-entropy loss.

* update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    Input Arguments:
        W1, b1, W2, b2 (NumPy Arrays): Current weights and biases.
        dW1, db1, dW2, db2 (NumPy Arrays): Gradients for weights and biases.
        learning_rate (Float): Learning rate for gradient descent.
    Output: None (In-place update).
    Description: Updates the weights and biases using gradient descent.

* train_neural_network(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
    Input Arguments:
        X_train, y_train (NumPy Arrays): Training data and labels.
        X_test, y_test (NumPy Arrays): Test data and labels.
        epochs (Integer): Number of training epochs.
        batch_size (Integer): Size of mini-batches.
        learning_rate (Float): Learning rate for gradient descent.
    Output: Trained weights and biases, and metrics for training.
    Description: Trains the neural network using mini-batch gradient descent.


Variables in function:
### Directory and File Handling

| Variable         | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| `extracted_dir`  | Specifies the directory where the MNIST dataset will be extracted.                                |
| `extracted_files`| List containing the names of all files in the `extracted_dir`.                                   |

---

### Loading MNIST Data

| Variable         | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| `train_images`   | Numpy array holding the training images.                                                         |
| `train_labels`   | Numpy array holding the training labels.                                                         |
| `test_images`    | Numpy array holding the test images.                                                             |
| `test_labels`    | Numpy array holding the test labels.                                                             |

---

### Neural Network Parameters Initialization

| Variable       | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `input_size`   | Number of input neurons (784 because MNIST images are 28x28).                                    |
| `hidden_size`  | Number of hidden neurons in the neural network.                                                  |
| `output_size`  | Number of output neurons (10 for the 10 digits).                                                 |
| `W1`           | Weights for the hidden layer.                                                                    |
| `b1`           | Biases for the hidden layer.                                                                     |
| `W2`           | Weights for the output layer.                                                                    |
| `b2`           | Biases for the output layer.                                                                     |

---

### Activation Functions

| Variable       | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `Z`            | Pre-activations (linear output) used in ReLU and Softmax functions.                              |
| `exp_Z`        | Exponentiated `Z` values, used in Softmax for numerical stability.                               |

---

### Forward Propagation

| Variable       | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `A1`           | Activations for the hidden layer.                                                                |
| `Z1`           | Pre-activations (linear output) for the hidden layer.                                            |
| `A2`           | Activations for the output layer.                                                                |
| `Z2`           | Pre-activations (linear output) for the output layer.                                            |
| `X_subset`     | Subset of training images used for testing forward propagation.                                  |
| `y_subset`     | Subset of training labels used for testing forward propagation.                                  |

---

### Backward Propagation

| Variable       | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `dA`           | Derivative of the loss with respect to the activated output `A`.                                 |
| `dZ`           | Derivative of the loss with respect to the linear output `Z`.                                    |
| `dW`           | Derivative of the loss with respect to the weights `W`.                                          |
| `db`           | Derivative of the loss with respect to the bias `b`.                                             |
| `dA_prev`      | Derivative of the loss with respect to the previous layer's activated output.                    |
| `y_encoded`    | One-hot encoded labels.                                                                          |
| `dA1`, `dA2`   | Specific derivatives of the loss for hidden and output layers, respectively.                     |

---

### Training Loop and Evaluation

| Variable             | Description                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------|
| `train_losses`       | List to store the loss values during training.                                                 |
| `test_accuracies`    | List to store the test accuracies during training.                                             |
| `epochs`             | Number of epochs for training.                                                                 |
| `batch_size`         | Size of each mini-batch.                                                                       |
| `learning_rate`      | Learning rate for gradient descent.                                                            |
| `W1_trained`         | Trained weights for the hidden layer.                                                          |
| `b1_trained`         | Trained biases for the hidden layer.                                                           |
| `W2_trained`         | Trained weights for the output layer.                                                          |
| `b2_trained`         | Trained biases for the output layer.                                                           |
| `shuffle_indices`    | Randomly shuffled indices used for shuffling the training data each epoch.                      |
| `X_train_shuffled`   | Shuffled training images for each epoch.                                                       |
| `y_train_shuffled`   | Shuffled training labels for each epoch.                                                       |
| `X_batch`            | Mini-batch of training images.                                                                  |
| `y_batch`            | Mini-batch of training labels.                                                                  |
| `loss`               | Computed loss value for each mini-batch.                                                       |
| `test_predictions`   | Predictions on the test set.                                                                    |
| `test_accuracy`      | Accuracy on the test set, calculated each epoch.                                                |




#### References/ helpful links:
[ChatGPT session for creation](https://chat.openai.com/share/0b8168ea-5d2a-497d-967e-c129e2424fcf) 

[ChatGPT session for creation of variable table and description](https://chat.openai.com/share/c0d0bf8c-bbf3-49b1-bdf6-6f6c71055f8c) 

https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch 