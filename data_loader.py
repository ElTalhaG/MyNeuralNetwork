import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    """Load MNIST data using scikit-learn"""
    print("Loading MNIST dataset from scikit-learn...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Split into train and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Convert labels to integers
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    return (X_train, y_train), (X_test, y_test)

def preprocess_data(X, y, num_classes=10):
    """Preprocess MNIST data"""
    # Normalize images
    X = X.astype('float32') / 255.0
    
    # Reshape for neural network input
    X = X.T
    
    # One-hot encode labels
    y_one_hot = np.zeros((num_classes, y.shape[0]))
    y_one_hot[y, np.arange(y.shape[0])] = 1
    
    return X, y_one_hot 