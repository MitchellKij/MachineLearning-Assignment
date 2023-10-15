# Initialize weights and biases
np.random.seed(0)  # For reproducibility

# Parameters for the hidden layer
input_size = 784  # Number of input neurons
hidden_size = 128  # Number of hidden neurons

W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for the hidden layer
b1 = np.zeros((1, hidden_size))  # Biases for the hidden layer

# Parameters for the output layer
output_size = 10  # Number of output neurons (10 classes) --> Corresponds to the 10 digits

W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for the output layer
b2 = np.zeros((1, output_size))  # Biases for the output layer

W1.shape, b1.shape, W2.shape, b2.shape

def relu(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)

def softmax(Z):
    """Softmax activation function"""
    exp_Z = np.exp(Z - np.max(Z))  # Numerical stability
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

# Forward propagation for a single layer
def forward_layer(A_prev, W, b, activation):
    """Forward propagation for a single layer"""
    Z = np.dot(A_prev, W) + b
    if activation == 'relu':
        A = relu(Z)
    elif activation == 'softmax':
        A = softmax(Z)
    return A, Z

# Forward propagation for the entire network
def forward_propagation(X, W1, b1, W2, b2):
    """Forward propagation for the entire network"""
    # Hidden layer
    A1, Z1 = forward_layer(X, W1, b1, 'relu')
    # Output layer
    A2, Z2 = forward_layer(A1, W2, b2, 'softmax')
    return A1, Z1, A2, Z2

# # Take a subset of training data
# X_subset = train_images[:5]
# y_subset = train_labels[:5]

# # Perform forward propagation
# A1, Z1, A2, Z2 = forward_propagation(X_subset, W1, b1, W2, b2)

# Backward propagation for a single layer
def backward_layer(dA, Z, A_prev, W, activation):
    """Backward propagation for a single layer"""
    if activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    elif activation == 'softmax':
        dZ = dA  # Softmax error is directly passed as dZ
    dW = np.dot(A_prev.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)
    return dA_prev, dW, db

# Backward propagation for the entire network
def backward_propagation(A1, Z1, A2, Z2, X, y, W2):
    """Backward propagation for the entire network"""
    m = X.shape[0]
    
    # Output layer
    y_encoded = one_hot_encode(y, 10) # One-hot encode the labels 10 classes
    dA2 = A2 - y_encoded 
    dA1, dW2, db2 = backward_layer(dA2, Z2, A1, W2, 'softmax') 
    
    # Hidden layer
    _, dW1, db1 = backward_layer(dA1, Z1, X, W1, 'relu')
    
    # Normalize gradients by the number of samples
    dW1 /= m
    db1 /= m
    dW2 /= m
    db2 /= m
    
    return dW1, db1, dW2, db2

# One-hot encode the labels
def one_hot_encode(labels, num_classes):
    """One-hot encode the given labels"""
    encoded = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

# Categorical Cross-Entropy Loss
def cross_entropy_loss(A, y):
    """Compute the categorical cross-entropy loss"""
    m = y.shape[0]
    y_encoded = one_hot_encode(y, 10)
    log_probs = np.log(A) * y_encoded
    loss = -np.sum(log_probs) / m
    return loss

# Update parameters using Gradient Descent
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """Update parameters using gradient descent"""
    W1 -= learning_rate * dW1 
    b1 -= learning_rate * db1 
    W2 -= learning_rate * dW2 
    b2 -= learning_rate * db2

# Training Loop
def train_neural_network(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
    """Train the neural network"""
    # Initialize parameters
    np.random.seed(0) # For reproducibility
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # Initialize weights for the hidden layer with small random values
    b1 = np.zeros((1, hidden_size))  # Initialize biases for the hidden layer as zeros

    W2 = np.random.randn(hidden_size, output_size) * 0.01  # Initialize weights for the output layer with small random values
    b2 = np.zeros((1, output_size))  # Initialize biases for the output layer as zeros
    
    # Store loss and accuracy for plotting
    train_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(1, epochs + 1): # Epoch loop
        # Shuffle training data for each epoch
        shuffle_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffle_indices]
        y_train_shuffled = y_train[shuffle_indices]
        
        for i in range(0, len(X_train), batch_size): # Mini-batch loop
            # Mini-batch data
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # Forward propagation
            A1, Z1, A2, Z2 = forward_propagation(X_batch, W1, b1, W2, b2)
            
            # Compute loss
            loss = cross_entropy_loss(A2, y_batch)
            
            # Backward propagation
            dW1, db1, dW2, db2 = backward_propagation(A1, Z1, A2, Z2, X_batch, y_batch, W2)
            
            # Update parameters
            update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        # Evaluate the model
        _, _, A2_test, _ = forward_propagation(X_test, W1, b1, W2, b2)
        test_predictions = np.argmax(A2_test, axis=1)
        test_accuracy = np.mean(test_predictions == y_test)
        
        # Store metrics
        train_losses.append(loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    return W1, b1, W2, b2, train_losses, test_accuracies

# Hyperparameters
epochs = 10
batch_size = 128 # Number of samples per mini-batch
learning_rate = 0.1 

# Train the neural network
W1_trained, b1_trained, W2_trained, b2_trained, train_losses, test_accuracies = train_neural_network(
    train_images, train_labels, test_images, test_labels, epochs, batch_size, learning_rate) # Train the Network
