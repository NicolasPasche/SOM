########## ¦ Imports ¦ ##########

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

########## ¦ Definitionen ¦ ##########

class Node(object):
    def __init__(self, input_size=2):
        self.input_size = input_size
        self.weight = np.array([random.random() for e in range(self.input_size)])

class SOM(object):
    def __init__(self, map_width=10, map_height=10, alpha=0.005, seed=1):
        '''
        Initialisierung der SOM
        '''
        self.map_width = map_width
        self.map_height = map_height
        self.alpha = alpha
        self.seed = seed
        self.radius = 0.6
        random.seed(self.seed)
        self.map = [[Node() for j in range(self.map_width)] for i in range(self.map_height)]

    def fit(self, iterations, X):
        '''
        Competitive learning für die SOM
        '''
        plot_iterations = {1, 2, 6, 100}  # Iterationen, in denen geplottet werden soll
        for s in range(1, iterations + 1):
            radius_s = self.radius * math.exp(-1.0 * s / iterations)
            alpha_s = self.alpha * (1.0 - s / iterations)
            x = X[random.randint(0, X.shape[0] - 1)]
            distances = np.empty((self.map_height, self.map_width))
            # Alle Distanzen berechnen
            for i in range(self.map_height):
                for j in range(self.map_width):
                    distances[i][j] = self.distance(x, self.map[i][j].weight)
            # Best Matching Unit Index finden
            bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
            for i in range(self.map_height):
                for j in range(self.map_width):
                    v = self.map[i][j].weight  # Gewichtsvektor eines Knotens
                    u = self.map[bmu_index[0]][bmu_index[1]].weight  # BMU
                    distance = self.distance(u, v)  # Distanz BMU und Gewichtsvektor
                    if distance <= radius_s:
                        neighborhood = radius_s
                        self.map[i][j].weight += neighborhood * alpha_s * (x - v)  # Gewichtsanpassung
            if s in plot_iterations:  # Plot für spezifische Iterationen
                self.plot_weights(x, s)

    def predict(self, y):
        '''
        Den nächsten Vektor berechnen
        '''
        distances = np.empty((self.map_height, self.map_width))
        for i in range(self.map_height):
            for j in range(self.map_width):
                distances[i][j] = self.distance(y, self.map[i][j].weight)
        # Knoten mit kleinstem Abstand zwischen Gewichtsvektor und Target y
        min_dist_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        # Rückgabe des Gewichtsvektors
        return self.map[min_dist_index[0]][min_dist_index[1]].weight

    def distance(self, u, v):
        '''
        Berechnung der Distanz zwischen zwei Vektoren
        '''
        return np.linalg.norm(u - v)

    def plot_weights(self, x, iteration):
        fig, ax = plt.subplots()
        weights_x = []
        weights_y = []
        for i in range(self.map_height):
            for j in range(self.map_width):
                weights_x.append(self.map[i][j].weight[0])
                weights_y.append(self.map[i][j].weight[1])
        ax.scatter(weights_x, weights_y, color='b')
        plt.title(f'Gewichtsvektoren - Iteration {iteration}')
        plt.xlabel('x')
        plt.ylabel('y')
        xticks = np.arange(0, 1, 0.1)
        yticks = np.arange(0, 1, 0.1)
        plt.yticks(yticks)
        plt.xticks(xticks)
        plt.show()

########## ¦ Main ¦ ##########

map_width = 4
map_height = 4
alpha = 0.8
iterations = 100
som = SOM(map_width, map_height, alpha)
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
print('Trainieren die XOR - Funktion')
som.fit(iterations, X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print('Aussagen zur XOR - Funktion')
print(f'Aussage 0 | 0, {som.predict([0, 0])}')
print(f'Aussage 0 | 1, {som.predict([0, 1])}')
print(f'Aussage 1 | 0, {som.predict([1, 0])}')
print(f'Aussage 1 | 1, {som.predict([1, 1])}')
