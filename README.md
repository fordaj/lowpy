# Welcome to Version 0.1!
The **Memristive Python Machine Learning** (MemPyML) package was built for researchers and chip designers alike. MemPyML seeks to streamline the investigation process when considering memristive devices for implementing a machine learning algorithm in hardware. By using the familiar keras syntax, it will be second nature to write GPU-optimized code to push your algorithm to its limits.
*Currently, all features are still a work in progress.*

# Features
The aim is to focus first on the algorithms most published on in the field of neuromorphic computing, for both static and time series datasets.
## Datasets
- MNIST
- Spoken MNIST
## Algorithms
- Single Layer Perceptron (SLP)
- Multi-Layer Perceprton (MLP)
- Long-Short Term Memory (LSTM)
- Convolutional Filters
- Max / Average Pooling
## Activation Functions
- Sigmoid
- ReLu
## Optimization Functions
- Stochastic Gradient Descent (SGD)
- SGD with Momentum
## Initialization Distributions
- Uniform 
- Normal
## Regularization Techniques
- Dropout
## Device Characteristics
- Write Variability
- Read Variability
- Random Telegraph Noise (RTN)
- Intermediate States
- Decay
- Directional Drift
- Bidirectional Drift


# Requirements
The following are required to use MemPyML:
- GPU: NVIDIA
- OS: Linux (should work on Windows, not tested)
- Python 3.0 or newer
- PyCUDA

