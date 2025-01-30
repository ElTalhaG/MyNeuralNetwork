import numpy as np
import pickle
import os

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.history = {'cost': [], 'accuracy': []}
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize weights and biases between layers"""
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2. / self.layer_sizes[l-1])
            self.parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
    
    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """Forward propagation step"""
        self.cache = {}
        self.cache['A0'] = X
        
        for l in range(1, len(self.layer_sizes)):
            Z = np.dot(self.parameters[f'W{l}'], self.cache[f'A{l-1}']) + self.parameters[f'b{l}']
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = self.sigmoid(Z)
        
        return self.cache[f'A{len(self.layer_sizes)-1}']
    
    def backward_propagation(self, X, Y):
        """Backward propagation step"""
        m = X.shape[1]
        L = len(self.layer_sizes) - 1
        
        # Initialize gradients
        gradients = {}
        
        # Output layer
        dZ = self.cache[f'A{L}'] - Y
        gradients[f'dW{L}'] = (1/m) * np.dot(dZ, self.cache[f'A{L-1}'].T)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA * self.sigmoid_derivative(self.cache[f'Z{l}'])
            gradients[f'dW{l}'] = (1/m) * np.dot(dZ, self.cache[f'A{l-1}'].T)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        """Update network parameters using gradients"""
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    def compute_accuracy(self, X, y):
        """Compute accuracy on given data"""
        predictions = self.predict(X)
        return np.mean(predictions == np.argmax(y, axis=0))
    
    def train(self, X, Y, X_val=None, Y_val=None, learning_rate=0.1, epochs=100):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            A = self.forward_propagation(X)
            
            # Compute cost
            m = Y.shape[1]
            cost = -(1/m) * np.sum(Y * np.log(A + 1e-15) + (1-Y) * np.log(1-A + 1e-15))
            
            # Compute accuracy
            accuracy = self.compute_accuracy(X, Y)
            
            # Store metrics
            self.history['cost'].append(cost)
            self.history['accuracy'].append(accuracy)
            
            # Backward propagation
            gradients = self.backward_propagation(X, Y)
            
            # Update parameters
            self.update_parameters(gradients, learning_rate)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")
                if X_val is not None and Y_val is not None:
                    val_accuracy = self.compute_accuracy(X_val, Y_val)
                    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        A = self.forward_propagation(X)
        return np.argmax(A, axis=0)
    
    def save_model(self, filepath):
        """Save the model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'layer_sizes': self.layer_sizes,
                'parameters': self.parameters,
                'history': self.history
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a model from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['layer_sizes'])
        model.parameters = data['parameters']
        model.history = data.get('history', {'cost': [], 'accuracy': []})
        return model
    
    def predict_image(self, image_path):
        """Predict digit from an image file"""
        from PIL import Image
        import numpy as np
        
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST format
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.reshape(784, 1)  # Reshape to (784, 1)
        img_array = img_array.astype('float32') / 255.0  # Normalize
        
        # Make prediction
        prediction = self.predict(img_array)
        return prediction[0] 