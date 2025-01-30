from neural_network import NeuralNetwork
from data_loader import load_mnist, preprocess_data
from drawing_board import DrawingBoard
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_history(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot cost
    plt.subplot(1, 2, 1)
    plt.plot(history['cost'])
    plt.title('Training Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_history.png')
    plt.close()

def train_model(epochs=200, learning_rate=0.05):
    """Train a new model with specified parameters"""
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    
    print("Preprocessing data...")
    X_train, y_train = preprocess_data(train_images, train_labels)
    X_test, y_test = preprocess_data(test_images, test_labels)
    
    print("Creating neural network...")
    # Larger network with two hidden layers
    layer_sizes = [784, 256, 128, 10]
    nn = NeuralNetwork(layer_sizes)
    
    print(f"Training neural network for {epochs} epochs...")
    nn.train(X_train, y_train, X_val=X_test, Y_val=y_test, 
            learning_rate=learning_rate, epochs=epochs)
    
    # Save the model
    model_path = 'models/mnist_model.pkl'
    nn.save_model(model_path)
    
    # Plot training history
    plot_training_history(nn.history)
    print("\nTraining history plot saved to 'plots/training_history.png'")
    
    # Test the network
    print("\nTesting neural network...")
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=0))
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    return nn

def predict_custom_image(model, image_path):
    """Predict digit from a custom image"""
    try:
        prediction = model.predict_image(image_path)
        print(f"Predicted digit: {prediction}")
        
        # Display the image
        plt.figure(figsize=(4, 4))
        plt.imshow(plt.imread(image_path), cmap='gray')
        plt.title(f'Predicted: {prediction}')
        plt.axis('off')
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error predicting image: {str(e)}")

def main():
    # Create model directory
    os.makedirs('models', exist_ok=True)
    model_path = 'models/mnist_model.pkl'
    
    # Interactive loop for predictions
    while True:
        print("\nOptions:")
        print("1. Draw and predict digit")
        print("2. Predict digit from image file")
        print("3. Train new model")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Load or train model if needed
            if os.path.exists(model_path):
                nn = NeuralNetwork.load_model(model_path)
            else:
                print("No model found. Training new model...")
                nn = train_model()
            
            # Create and run drawing board
            board = DrawingBoard()
            board.run(nn)
            
        elif choice == '2':
            # Load or train model if needed
            if os.path.exists(model_path):
                nn = NeuralNetwork.load_model(model_path)
            else:
                print("No model found. Training new model...")
                nn = train_model()
                
            image_path = input("Enter the path to your image file: ")
            predict_custom_image(nn, image_path)
            
        elif choice == '3':
            # Get training parameters
            try:
                epochs = int(input("Enter number of epochs (default: 200): ") or "200")
                learning_rate = float(input("Enter learning rate (default: 0.05): ") or "0.05")
                print("\nStarting training...")
                nn = train_model(epochs=epochs, learning_rate=learning_rate)
            except ValueError:
                print("Invalid input. Using default values...")
                nn = train_model()
            
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 