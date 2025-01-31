# Self-Organizing Map (SOM) Implementation

This repository contains an implementation of a simple Self-Organizing Map (SOM) in Python. The SOM is an unsupervised learning algorithm that maps high-dimensional input data onto a lower-dimensional grid, where similar data points are placed closer together.

## Features

- Customizable map size with adjustable width and height
- Implements competitive learning for weight adjustments based on the Best Matching Unit (BMU)
- Supports dynamic learning rate (`alpha`)
- Visualizes the weights of the map during specific training iterations
- Random initialization of weihghts with reproducible results using a random seed

## Requirements

- `numpy`: For numerical computations and distance calculations
- `matplotlib`: For plotting the weight updates during training

You can install these dependencies using pip:
```bash
pip install numpy matplotlib
```

## Usage

The main class in the implementation is `SOM`. It provides methods to define, train and evaluate the self-organizing map.

### 1. Initialization

You can initialize the SOM with the desired map dimensions and learning parameters. The weights are initialized randomly.
```python
som = SOM(map_width=4, map_height=4, alpha=0.8, iterations=100)
```

### Training
The `fit` method trains the map using the provided input data (`X`). The map's weights are updated over the specified number of iterations.
```python
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
som.fit(iterations=100, X=X)
```

### 3. Prediction

Use the `predict` method to find the Best Matching Unit (BMU) for new input data. The method returns the weights of the node that best matches the input data.
```python
prediction = som.predict([1.0, 1.0])
```

### 4. Visualisazion

The `plot_weights` method generates a plot of the weight vectors at specific iterations during training. This helps visualize the map's development over time.
```python
som.plot_weights(x, iteration)
```

## Example
Here is a complete example:
```python
import numpy as np
from som import SOM

# Define training data
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

# Initialize the SOM
som = SOM(map_width=4, map_height=4, alpha=0.8, iterations=100)

# Train the SOM
som.fit(iterations=100, X=X)

# Print predictions for each input
print('Predictions for XOR function:')
for x in X:
    print(f'Input: {x}, Prediction: {som.predict(x)}')

# Plot the weights at specific iterations
som.plot_weights(x, iteration=100)
```

## Code Overview

### SOM Class

The `SOM` class is the core of this implementation. Below are its key components:

**Initialization (init):**
- Sets up map dimensions (width and height)
- Initializes the weights randomly
- Configures learning parameters (e.g. learning rate, rdius, number of iterations)

**Training (fit)**
- Updates the weights of the map nodes based on competitive learning
- Adjusts the weights using the Best Matching Unit (BMU) and the learning rate
- Dynamically adjusts the neighborood radius and learning rate over iterations

**Prediction (predict):**
- Computes the BMU for new input data by calculating the distances to each map node
- Returns the weight vector corresponding to the BMU

**Visualization (plot_weights):**
- Visualizes the weight vectors of the map at specific iterations during training
- Plots the development of the map as the algorithm progresses

## Limitations

- The implementation only suppeots a 2D input space for visualization (X and Y dimensions)
- The map's performance may degrade on very high-dimensional data due to the competitive learning nature of SOMs
- The current implementation doesn't include advanced techniques such as batch training

## Future Improvements

- Add support for higher-dimensional input data visualization
- Implement the use of different neighborhood functions (e.g. Gaussian)
- Optimize the algorithm for faster convergence
- Add the ability to save and load trained SOM models

## Let's connect!

If you have any questions, propositions or are looking to cooperate, feel free to contact me at nicolas.pasche@proton.me

## License

This project is licensed under the MIT License. Feel free to use, modify and distribute it as per the terms of the license.
