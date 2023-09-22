# Learning Journal 


Flattening the Images: Each image in the MNIST dataset is a 2D array (28x28 pixels). For simplicity, 2D array is flattened into a 1D array of 784 numbers. 

$\text{Flattened Image} \in \mathbb{R}^{784}$

The pixel values for each image in the dataset are grayscale values ranging from 0 to 255. We normalize these values to the range [0, 1] by dividing each pixel value by 255. This helps in faster and more stable convergence during the training process.

$\text{Normalized Pixel Value} = \frac{\text{Original Pixel Value}}{255}$

## Neural Network implimentation
**Forward propagation** is the initial phase of the two-step process involved in training a neural network. In this step, the input is passed through each layer of the network to produce an output. The operations involved are:

    Linear Transformation: This operation is represented mathematically as XW+b, where X is the input, W is the weight matrix, and bb is the bias vector. This step transforms the input into a form that can be activated by the subsequent activation function.
$Z = XW + b $

    Activation Function: After the linear transformation, the resulting value is passed through an activation function. We used ReLU (Rectified Linear Unit) for the hidden layer and Softmax for the output layer.
$\text{ReLU}(x) = \max(0, x) $

$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $




#### Citation:
[ChatGPT session](https://chat.openai.com/share/0b8168ea-5d2a-497d-967e-c129e2424fcf) 
