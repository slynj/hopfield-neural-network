import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """Init weight matrix.

        Args:
            size (int): # of neurons in the network. Determines the dimension of the weight matrix.
        """
        self.size = size
        self.weights = np.zeros((size, size))
    

    def train(self, patterns):
        """Training with Hebbian learning rule. 

        Updates the weight matrix based on the patterns. It computes the outer product of each 
        pattern with itself and accumulates the result to the weight matrix. Weight with itself
        is 0, and the matrix is normalized.

        Args:
            patterns (array-like): A list or array of patterns for training. Pattern should be
            a array-like struct (list, numpy array, etc) with bipolar values corresponding to
            neuron states.
        """
        for p in patterns:
            p = np.array(p)
            # computes the outer product of vector p and p
            #  => outer product of 2 vectors = a * b transposed
            #  => square matrix, each elem is the product of 2 elements from p
            #     outerP[i][j] = p[i] * p[j]
            self.weights += np.outer(p, p)
        
        np.fill_diagonal(self.weights, 0) # w_{ii} = 0

        # normalization step => bc we keep adding the outer product
        self.weights /= len(patterns) 
    

    def sign(self, x):
        """Activation Function, updates the neuron's state.

        Args:
            x (numpy.ndarray or float): Input value or array of values to be transformed.

        Returns:
            numpy.ndarray or int: Transformed input where each element is -1 or 1. Returns a single int if input is a scaler.
        """
        # ternery => convert nums to bipolar
        return np.where(x >= 0, 1, -1)