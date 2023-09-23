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

## Neural Network implimentation
**Forward propagation** is the initial phase of the two-step process involved in training a neural network. In this step, the input is passed through each layer of the network to produce an output. The operations involved are:

Linear Transformation: This operation is represented mathematically as $Z = XW + b $, where $X$ is the input, $W$ is the weight matrix, and $b$ is the bias vector. This step transforms the input into a form that can be activated by the subsequent activation function.



**Activation Functions:** After the linear transformation, the resulting value is passed through an activation function. We used ReLU (Rectified Linear Unit) is used for the hidden layer and Softmax for the output layer.
$\text{ReLU}(x) = \max(0, x) $

$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $

## Forward Propagation

Forward propagation is the initial phase of the learning process. During this phase, the neural network makes its initial guesses about the output.

Linear Transformation: For each layer, the input $A_{prev}$$​ is linearly transformed using the weights $W$ and biases $b$ to produce $Z$.

Activation: The linearly transformed input $Z$ is then passed through an activation function to introduce non-linearity, resulting in $A$.

Output: The final layer's activation output is the network's prediction, which is then compared to the actual label to compute the loss.

## Backward Propagation

Backward propagation is the stage where the neural network learns from its mistakes. The aim is to minimize the loss function.

Compute Gradients: The first step in backpropagation is to compute the gradient of the loss function with respect to each weight and bias by applying the chain rule of calculus.

Update Parameters: The gradients calculated are then used to update the weights and biases in the network. The code uses gradient descent

Error Propagation: The 'backward' in backpropagation refers to the fact that the calculation of the gradients starts from the output layer and moves backward through the network, allowing for the efficient computation of gradients.

Normalization: In the code, gradients are normalized by the number of samples to ensure that the weight updates are not influenced by the size of the data batch.

## Gradient Descent
Gradient Descent: This is one of the most basic optimization algorithms used for finding the minimum of a function. In the context of neural networks, this function is the loss function. Gradient Descent updates each parameter $θ$ according to the rule:

$θ=θ−α∇J(θ)$

where $α$ is the learning rate and $∇J(θ)$ is the gradient of the loss function $J$ with respect to $θ$.

Learning Rate ($α$): This is a hyperparameter that controls the size of the updates to the parameters. Too large a learning rate might cause the model to converge too quickly and overshoot the minimum, while too small a learning rate could make the model very slow to converge or get stuck in a local minimum.

Gradients:  These are the partial derivatives of the loss function with respect to each parameter. They are computed during the backward propagation phase and indicate the direction and magnitude by which each parameter should be updated.

Parameter Update: The actual parameters W1,b1,W2,b2 are updated in the negative direction of the gradient. This is based on the principle that the function decreases fastest if one goes in the direction of the negative gradient.

#### References:
[ChatGPT session](https://chat.openai.com/share/0b8168ea-5d2a-497d-967e-c129e2424fcf) 
