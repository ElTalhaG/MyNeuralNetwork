# MNIST Digit Recognition Neural Network

This project implements a neural network from scratch to recognize handwritten digits from the MNIST dataset.

## Project Structure
- `neural_network.py`: Contains the neural network implementation
- `data_loader.py`: Handles downloading and preprocessing MNIST data
- `main.py`: Main script to train and test the neural network
- `requirements.txt`: List of Python dependencies

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
Simply run the main script:
```bash
python main.py
```

The script will:
1. Download the MNIST dataset (if not already present)
2. Preprocess the data
3. Train the neural network
4. Test its accuracy on the test set

## Neural Network Architecture
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: 128 neurons with sigmoid activation
- Output layer: 10 neurons (one for each digit 0-9)

## Features
- Built from scratch using NumPy
- Implements forward and backward propagation
- Uses sigmoid activation function
- Includes proper initialization of weights and biases
- Automatically downloads and processes MNIST data 