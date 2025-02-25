import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        for p in patterns:
            p = np.array(p)
            # computes the outer product of vector p and p
            #  => outer product of 2 vectors = a * b transposed
            #  => square matrix, each elem is the product of 2 elements from p
            #     outerP[i][j] = p[i] * p[j]
            self.weights += np.outer(p, p)